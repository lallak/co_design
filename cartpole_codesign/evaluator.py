import sys
from pathlib import Path
from functools import lru_cache

hydrax_parent = Path(__file__).parent / "hydrax"
sys.path.insert(0, str(hydrax_parent))

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.algs import MPPI
import matplotlib.pyplot as plt
import os

from cartpole_codesign.tasks.cartpole_task import CartpoleSwingupTask
from cartpole_codesign.assets.model_builder import build_cartpole_model

# Hyperparameters (fixed across all designs)
PLAN_HORIZON  = 1.0
NUM_KNOTS     = 10
DT            = 0.01
NUM_SAMPLES   = 1024
NOISE_LEVEL   = 0.5
TEMPERATURE   = 0.5
EPISODE_STEPS = 400
NUM_EPISODES  = 3

POLE_MASS_FIXED = 0.1  # kg (fixed, not optimized)


@lru_cache(maxsize=256)
def _build_controller(theta_tuple):
    """Construit et JIT-compile le controller pour un design donné."""
    theta_full = np.array(theta_tuple)
    mj_model   = build_cartpole_model(theta_full)
    mjx_model  = mjx.put_model(mj_model)          # ← modèle GPU pour mjx.step
    task       = CartpoleSwingupTask(mj_model)
    controller = MPPI(
        task,
        num_samples=NUM_SAMPLES,
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=PLAN_HORIZON,
        num_knots=NUM_KNOTS,
    )
    jit_optimize   = jax.jit(controller.optimize)
    jit_get_action = jax.jit(controller.get_action)
    return mj_model, mjx_model, task, controller, jit_optimize, jit_get_action  # ← mjx_model ajouté


def evaluate_design(theta: np.ndarray, seed: int = 0) -> float:
    """
    Évalue un design en lançant MPPI sur CartPole swingup.

    Args:
        theta: [pole_length, cart_mass]
        seed:  graine aléatoire

    Returns:
        mean_reward: récompense moyenne sur NUM_EPISODES épisodes (plus haut = meilleur).
    """
    pole_length, cart_mass = theta[0], theta[1]
    theta_full = np.array([pole_length, POLE_MASS_FIXED, cart_mass])
    theta_key  = tuple(np.round(theta_full, 4))

    mj_model, mjx_model, task, controller, jit_optimize, jit_get_action = _build_controller(theta_key)  # ← mjx_model déballé

    # État initial CPU — uploadé une seule fois avant la boucle GPU
    mj_data_init = mujoco.MjData(mj_model)
    mj_data_init.qpos[1] = np.pi
    mj_data_init.qvel[:] = 0.0

    def run_single_episode(episode_seed_key):
        """Lance un épisode complet sur GPU. Vectorisable via vmap."""
        perturb_key, _ = jax.random.split(episode_seed_key)
        perturb = jax.random.uniform(perturb_key, shape=(), minval=-0.1, maxval=0.1)

        # Upload unique CPU → GPU
        mjx_data = mjx.put_data(mj_model, mj_data_init)
        new_qpos = mjx_data.qpos.at[1].add(perturb)
        mjx_data = mjx_data.replace(qpos=new_qpos)

        initial_knots = jnp.zeros((NUM_KNOTS, mj_model.nu), dtype=jnp.float32)
        ctrl_state = controller.init_params(initial_knots=initial_knots, seed=0)

        def step_fn(carry, _):
            mjx_data, ctrl_state = carry
            ctrl_state, _rollouts = jit_optimize(mjx_data, ctrl_state)
            action   = jit_get_action(ctrl_state, mjx_data.time)
            mjx_data = mjx_data.replace(ctrl=action)
            mjx_data = mjx.step(mjx_model, mjx_data)   # ← mjx_model (GPU), pas mj_model (CPU)
            cost     = task.running_cost(mjx_data, action)
            return (mjx_data, ctrl_state), cost

        (_final_data, _final_ctrl), costs = jax.lax.scan(
            step_fn,
            (mjx_data, ctrl_state),
            None,
            length=EPISODE_STEPS,
        )
        return -jnp.sum(costs)

    rng          = jax.random.PRNGKey(seed)
    episode_keys = jax.random.split(rng, NUM_EPISODES)
    rewards      = jax.jit(jax.vmap(run_single_episode))(episode_keys)
    return float(jnp.mean(rewards))


def generate_heatmap_2d(
    pole_length_range=[0.2, 1.3],
    cart_mass_range=[0.1, 2.5],
    resolution=15,
    output_path="results_cartpole/results_cartpole/heatmap_2d_20x20_shifted.png",
):
    pole_lengths = np.linspace(pole_length_range[0], pole_length_range[1], resolution)
    cart_masses  = np.linspace(cart_mass_range[0],  cart_mass_range[1],  resolution)
    heatmap      = np.zeros((len(cart_masses), len(pole_lengths)))

    print(f"Generating {resolution}x{resolution} heatmap (pole_mass={POLE_MASS_FIXED} kg fixed)...")

    for i, cart_mass in enumerate(cart_masses):
        for j, pole_length in enumerate(pole_lengths):
            theta  = np.array([pole_length, cart_mass])
            reward = evaluate_design(theta, seed=i * resolution + j)
            heatmap[i, j] = reward
            print(f"  [{i+1}/{len(cart_masses)}, {j+1}/{len(pole_lengths)}] "
                  f"pole_length={pole_length:.2f}, cart_mass={cart_mass:.2f} → reward={reward:.3f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, origin="lower",
                   extent=[pole_lengths.min(), pole_lengths.max(),
                           cart_masses.min(),  cart_masses.max()],
                   aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_xlabel("Pole Length (m)", fontsize=12)
    ax.set_ylabel("Cart Mass (kg)", fontsize=12)
    ax.set_title(f"CartPole Design Landscape\n(pole_mass={POLE_MASS_FIXED} kg fixed)", fontsize=14)
    plt.colorbar(im, ax=ax, label="Reward")

    best_idx         = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    best_pole_length = pole_lengths[best_idx[1]]
    best_cart_mass   = cart_masses[best_idx[0]]
    best_reward      = heatmap[best_idx]
    ax.plot(best_pole_length, best_cart_mass, "r*", markersize=25,
            label=f"Best: {best_reward:.3f}", markeredgecolor="black", markeredgewidth=1.5)
    ax.legend(fontsize=11, loc="upper left")

    contours = ax.contour(pole_lengths, cart_masses, heatmap,
                          levels=5, colors="black", alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved to {output_path}")
    print(f"\n=== Heatmap Summary ===")
    print(f"  pole_length = {best_pole_length:.4f} m")
    print(f"  cart_mass   = {best_cart_mass:.4f} kg")
    print(f"  pole_mass   = {POLE_MASS_FIXED:.4f} kg (fixed)")
    print(f"  Best reward = {best_reward:.4f}")

    np.savez(output_path.replace(".png", "_data.npz"),
             pole_lengths=pole_lengths, cart_masses=cart_masses,
             heatmap=heatmap, pole_mass_fixed=POLE_MASS_FIXED)
    return heatmap, pole_lengths, cart_masses

'''import sys
from pathlib import Path

from jax import lax

# Add hydrax directory to path so hydrax imports work
hydrax_parent = Path(__file__).parent / "hydrax"
sys.path.insert(0, str(hydrax_parent))

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.algs import MPPI
import matplotlib.pyplot as plt
import os

from cartpole_codesign.tasks.cartpole_task import CartpoleSwingupTask
from cartpole_codesign.assets.model_builder import build_cartpole_model

# Hyperparameters (fixed across all designs)
PLAN_HORIZON = 1.0      # seconds
NUM_KNOTS    = 10        # spline control points (MPPI default)
DT           = 0.01     # simulation timestep
NUM_SAMPLES  = 256     # MPPI trajectory samples per step
NOISE_LEVEL  = 0.5      # MPPI exploration noise
TEMPERATURE = 0.5
EPISODE_STEPS = 400     # steps per episode (~4 seconds)
NUM_EPISODES  = 3     # episodes per design evaluation

# --- FAST MODE: evaluator ---
PLAN_HORIZON = 1.0
NUM_KNOTS    = 4
DT           = 0.01  # simulation timestep
NUM_SAMPLES  = 128 #try more than 64
NOISE_LEVEL  = 0.5
TEMPERATURE = 0.5
EPISODE_STEPS = 200
NUM_EPISODES  = 1

# Fixed parameter
POLE_MASS_FIXED = 0.1   # kg (fixed, not optimized)


def evaluate_design(theta: np.ndarray, seed: int = 0) -> float:
    """
    Evaluate a design by running MPPI on the CartPole swingup task.

    Args:
        theta: Design vector [pole_length, cart_mass]
               (pole_mass is fixed at POLE_MASS_FIXED)
        seed:  Random seed for reproducibility

    Returns:
        mean_reward: Average cumulative reward across NUM_EPISODES episodes.
                     Higher is better (we negate this for CMA-ES which minimizes).
    """
    # Extract 2 parameters to optimize
    pole_length = theta[0]
    cart_mass = theta[1]
    pole_mass = POLE_MASS_FIXED  # fixed parameter

    # Create theta for model_builder (3 parameters)
    theta_full = np.array([pole_length, pole_mass, cart_mass])

    # --- Build model and task ---
    mj_model = build_cartpole_model(theta_full)
    task = CartpoleSwingupTask(mj_model)

    # --- Initialize MPPI controller ---
    controller = MPPI(task,
                      num_samples=NUM_SAMPLES,
                      noise_level=NOISE_LEVEL,
                      temperature=TEMPERATURE,
                      plan_horizon=PLAN_HORIZON,
                      num_knots=NUM_KNOTS) #control points used to parameterize the action sequence in MPPI. More knots = more complex plans

    # --- JIT-compile the MPC update step (do this once per design) ---
    jit_update = jax.jit(controller.optimize)
    jit_get_action = jax.jit(controller.get_action)

    # --- Run episodes ---
    rng = jax.random.PRNGKey(seed)
    total_reward = 0.0

    for episode in range(NUM_EPISODES):
        # Reset to initial state: pole hanging down, small random perturbation
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[1] = np.pi + np.random.uniform(-0.1, 0.1)  # pole down
        mj_data.qvel[:] = 0.0

        # Convert to MJX data for JAX operations
        mjx_data = mjx.put_data(mj_model, mj_data)

        # Initialize controller params with the correct knot shape
        initial_knots = np.zeros((NUM_KNOTS, mj_model.nu), dtype=np.float32)
        ctrl_state = controller.init_params(initial_knots=initial_knots, seed=seed + episode)

        episode_reward = 0.0

        for step in range(EPISODE_STEPS): #basically MPPI steps : trajectories, optimal action, step sim, accumulate reward
            # MPC update: sample trajectories and compute optimal action
            rng, subkey = jax.random.split(rng)
            ctrl_state, rollouts = jit_update(mjx_data, ctrl_state)

            # Get the first action from the optimal plan
            action = jit_get_action(ctrl_state, mjx_data.time)

            # Apply action and step simulation
            mj_data.ctrl[:] = np.array(action)
            mjx_data = mjx.put_data(mj_model, mj_data)

            # Accumulate reward (negative cost)
            cost = task.running_cost(mjx_data, action)
            episode_reward -= float(cost)

        total_reward += episode_reward

    return total_reward / NUM_EPISODES

def generate_heatmap_2d(pole_length_range=[0.3, 1.5], #try less pole lent in the range
                        cart_mass_range=[0.5, 3.0],
                        resolution=15,
                        output_path="results_cartpole/heatmap_2d_20x20_shifted.png"):
    """
    Generate a 2D heatmap of rewards vs pole_length and cart_mass.
    
    Args:
        pole_length_range: [min, max] for pole_length axis
        cart_mass_range: [min, max] for cart_mass axis
        resolution: Number of grid points per axis (resolution x resolution grid)
        output_path: Where to save the plot
    """
    import matplotlib.pyplot as plt
    
    # Create grid
    pole_lengths = np.linspace(pole_length_range[0], pole_length_range[1], resolution)
    cart_masses = np.linspace(cart_mass_range[0], cart_mass_range[1], resolution)
    
    # Evaluate on grid
    heatmap = np.zeros((len(cart_masses), len(pole_lengths)))
    
    print(f"Generating {resolution}x{resolution} heatmap (pole_mass={POLE_MASS_FIXED} kg fixed)...")
    
    for i, cart_mass in enumerate(cart_masses):
        for j, pole_length in enumerate(pole_lengths):
            theta = np.array([pole_length, cart_mass])
            reward = evaluate_design(theta, seed=i*resolution + j)
            heatmap[i, j] = reward
            print(f"  [{i+1}/{len(cart_masses)}, {j+1}/{len(pole_lengths)}] "
                  f"pole_length={pole_length:.2f}, cart_mass={cart_mass:.2f} → reward={reward:.3f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(heatmap, 
                   origin='lower',
                   extent=[pole_lengths.min(), pole_lengths.max(),
                          cart_masses.min(), cart_masses.max()],
                   aspect='auto',
                   cmap='RdYlGn',  # Red=bad, Yellow=medium, Green=good
                   interpolation='nearest')
    
    ax.set_xlabel('Pole Length (m)', fontsize=12)
    ax.set_ylabel('Cart Mass (kg)', fontsize=12)
    ax.set_title(f'CartPole Design Landscape\n(pole_mass={POLE_MASS_FIXED} kg fixed)', 
                 fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, label='Reward')
    
    # Mark the best point
    best_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    best_pole_length = pole_lengths[best_idx[1]]
    best_cart_mass = cart_masses[best_idx[0]]
    best_reward = heatmap[best_idx]
    
    ax.plot(best_pole_length, best_cart_mass, 'r*', markersize=25, 
            label=f'Best: {best_reward:.3f}', markeredgecolor='black', markeredgewidth=1.5)
    ax.legend(fontsize=11, loc='upper left')
    
    # Add contour lines
    contours = ax.contour(pole_lengths, cart_masses, heatmap, levels=5, colors='black', 
                         alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nHeatmap saved to {output_path}")
    
    # Print summary
    print(f"\n=== Heatmap Summary ===")
    print(f"Best design:")
    print(f"  pole_length = {best_pole_length:.4f} m")
    print(f"  cart_mass   = {best_cart_mass:.4f} kg")
    print(f"  pole_mass   = {POLE_MASS_FIXED:.4f} kg (fixed)")
    print(f"  Best reward = {best_reward:.4f}")
    
    # Save data for later analysis
    np.savez(output_path.replace('.png', '_data.npz'),
             pole_lengths=pole_lengths,
             cart_masses=cart_masses,
             heatmap=heatmap,
             pole_mass_fixed=POLE_MASS_FIXED)
    
    return heatmap, pole_lengths, cart_masses'''