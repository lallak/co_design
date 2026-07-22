import os
import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from matplotlib import pyplot as plt
from mujoco import mjx
from hydrax.algs import MPPI
from hydrax.algs import DIAL

from hopper_codesign.tasks.hopper_task import HopperLocomotionTask
from hopper_codesign.assets.model_builder import build_hopper_model

# Hyperparameters
PLAN_HORIZON = 1.0 #duration that MPC plans into the future
NUM_KNOTS = 10 #number of control points used to represent control trajectory
# More knots allow more complex motions but increase the optimization dimension.
NUM_SAMPLES = 1024 #number of trajectories sampled at each MPC update
NOISE_LEVEL = 0.5 #std deviation of gaussian noise added to sampled trajectories
# Larger values encourage exploration, smaller values focus on refinement.
TEMPERATURE = 0.5 #controls how much best trajectories influence final control (in weighted average calculation)
# Lower values make the optimizer greedier; higher values average over more samples.
EPISODE_STEPS = 400 # Number of simulation steps in one evaluation episode.
NUM_EPISODES = 6 # Number of episodes used to evaluate each design.
BETA_OPT_ITER = 0.8 # DIAL parameter controlling interpolation between exploration and exploitation
#the higher the quicker the controller focuses on promising regions
BETA_HORIZON = 0.8 # DIAL parameter controlling how exploration changes along the planning horizon.
#Earlier controls are more refined than late controls

'Basically exploration when the kernel (noise) is high and exploitation when focusing on the promising trajectories with a low kernel.'

@lru_cache(maxsize=256)
def _build_controller(theta_tuple):
    """Build and JIT-compile the controller."""
    theta_full = np.array(theta_tuple)
    mj_model = build_hopper_model(theta_full)
    mjx_model = mjx.put_model(mj_model)
    task = HopperLocomotionTask(mj_model)
    """controller = MPPI(
        task,
        num_samples=NUM_SAMPLES,
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=PLAN_HORIZON,
        num_knots=NUM_KNOTS,
    )"""

    controller = DIAL(
        task,
        num_samples=NUM_SAMPLES,
        noise_level=NOISE_LEVEL,
        beta_opt_iter=BETA_OPT_ITER,
        beta_horizon=BETA_HORIZON,
        temperature=TEMPERATURE,
        plan_horizon=PLAN_HORIZON,
        num_knots=NUM_KNOTS,
    )
    jit_optimize = jax.jit(controller.optimize)
    jit_get_action = jax.jit(controller.get_action)
    return mj_model, mjx_model, task, controller, jit_optimize, jit_get_action


def evaluate_design(theta: np.ndarray, seed: int = 0) -> float:
    """Evaluate a hopper design."""
    theta_key = tuple(np.round(theta, 4))
    mj_model, mjx_model, task, controller, jit_optimize, jit_get_action = _build_controller(theta_key)

    mj_data_init = mujoco.MjData(mj_model)
    mj_data_init.qpos[1] = 0.1
    mj_data_init.qvel[:] = 0.0

    def run_single_episode(episode_seed_key):
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
            ctrl_state, _ = jit_optimize(mjx_data, ctrl_state)
            action = jit_get_action(ctrl_state, mjx_data.time)
            mjx_data = mjx_data.replace(ctrl=action)
            mjx_data = mjx.step(mjx_model, mjx_data)
            cost = task.running_cost(mjx_data, action)
            return (mjx_data, ctrl_state), cost

        (final_data, _), costs = jax.lax.scan(
            step_fn,
            (mjx_data, ctrl_state),
            None,
            length=EPISODE_STEPS,
        )

        distance_traveled = final_data.qpos[0]
        return distance_traveled - jnp.sum(costs)

    rng = jax.random.PRNGKey(seed)
    episode_keys = jax.random.split(rng, NUM_EPISODES)
    rewards = jax.jit(jax.vmap(run_single_episode))(episode_keys)
    return float(jnp.mean(rewards))


def debug_controller(theta):
    theta_key = tuple(np.round(theta, 4))
    mj_model, mjx_model, task, controller, \
        jit_optimize, jit_get_action = _build_controller(theta_key)

    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[1] = 1.0
    mj_data.qvel[:] = 0.0

    mjx_data = mjx.put_data(mj_model, mj_data)
    initial_knots = jnp.zeros((NUM_KNOTS, mj_model.nu), dtype=jnp.float32)
    ctrl_state = controller.init_params(initial_knots=initial_knots, seed=0)

    heights, angles, forward_vels = [], [], []
    thigh_actions, leg_actions, foot_actions = [], [], []
    rewards = []

    for step in range(EPISODE_STEPS):
        ctrl_state, _ = jit_optimize(mjx_data, ctrl_state)
        action = jit_get_action(ctrl_state, mjx_data.time)
        mjx_data = mjx_data.replace(ctrl=action)
        mjx_data = mjx.step(mjx_model, mjx_data)
        reward = -task.running_cost(mjx_data, action)

        heights.append(float(mjx_data.qpos[1]))
        angles.append(float(mjx_data.qpos[2]))
        forward_vels.append(float(mjx_data.qvel[0]))
        thigh_actions.append(float(action[0]))
        leg_actions.append(float(action[1]))
        foot_actions.append(float(action[2]))
        rewards.append(float(reward))

    print("Final distance =", float(mjx_data.qpos[0]))
    print("Final height   =", float(mjx_data.qpos[1]))
    print("Final angle    =", float(mjx_data.qpos[2]))

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(heights);
    axs[0].set_title("Torso Height");
    axs[0].grid(True)
    axs[1].plot(angles);
    axs[1].set_title("Torso Angle");
    axs[1].grid(True)
    axs[2].plot(forward_vels);
    axs[2].set_title("Forward Velocity");
    axs[2].grid(True)
    plt.tight_layout();
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(thigh_actions, label="thigh")
    plt.plot(leg_actions, label="leg")
    plt.plot(foot_actions, label="foot")
    plt.title("Motor Commands");
    plt.xlabel("Step");
    plt.ylabel("Action")
    plt.legend();
    plt.grid(True);
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(rewards);
    plt.title("Reward");
    plt.grid(True);
    plt.show()


def sensitivity_analysis(nominal_theta: np.ndarray, n_eval: int = 5, delta: float = 0.1) -> dict:
    """Measure reward sensitivity to each design parameter."""
    param_names = ['thigh_length', 'leg_length', 'rho']

    base_reward = evaluate_design(nominal_theta, seed=0)
    print(f"Base reward: {base_reward:.2f}")

    sensitivities = {}
    for i, name in enumerate(param_names):
        perturbed = nominal_theta.copy()
        perturbed[i] *= (1 + delta)
        perturbed_reward = np.mean([evaluate_design(perturbed, seed=s) for s in range(n_eval)])
        sensitivities[name] = abs(perturbed_reward - base_reward)
        print(f"  {name:>15}: Δreward = {sensitivities[name]:.3f}")

    return sensitivities


RHO_FIXED = 2.1  # converged value

def generate_heatmap_2d(
    thigh_length_range = [0.15, 0.35],  # tight around converged ~0.28
    leg_length_range = [0.30, 0.70],  # full range since it didn't converge
    resolution=10,
    output_path="results/results_hopper/heatmap_2d_hopper_10x10_DIAL_10knots.png",
):
    thigh_lengths = np.linspace(thigh_length_range[0], thigh_length_range[1], resolution)
    leg_lengths   = np.linspace(leg_length_range[0],   leg_length_range[1],   resolution)
    heatmap       = np.zeros((len(leg_lengths), len(thigh_lengths)))

    print(f"Generating {resolution}x{resolution} heatmap (rho={RHO_FIXED} kg/m fixed)...")

    for i, leg_length in enumerate(leg_lengths):
        for j, thigh_length in enumerate(thigh_lengths):
            theta  = np.array([thigh_length, leg_length, RHO_FIXED])
            reward = evaluate_design(theta, seed=i * resolution + j)
            heatmap[i, j] = reward
            print(f"  [{i+1}/{len(leg_lengths)}, {j+1}/{len(thigh_lengths)}] "
                  f"thigh={thigh_length:.2f}, leg={leg_length:.2f} → reward={reward:.3f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, origin="lower",
                   extent=[thigh_lengths.min(), thigh_lengths.max(),
                           leg_lengths.min(),   leg_lengths.max()],
                   aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_xlabel("Thigh Length (m)", fontsize=12)
    ax.set_ylabel("Leg Length (m)",   fontsize=12)
    ax.set_title(f"Hopper Design Landscape\n(rho={RHO_FIXED} kg/m fixed)", fontsize=14)
    plt.colorbar(im, ax=ax, label="Reward")

    best_idx          = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    best_thigh_length = thigh_lengths[best_idx[1]]
    best_leg_length   = leg_lengths[best_idx[0]]
    best_reward       = heatmap[best_idx]
    ax.plot(best_thigh_length, best_leg_length, "r*", markersize=25,
            label=f"Best: {best_reward:.3f}", markeredgecolor="black", markeredgewidth=1.5)
    ax.legend(fontsize=11, loc="upper left")

    contours = ax.contour(thigh_lengths, leg_lengths, heatmap,
                          levels=5, colors="black", alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved to {output_path}")
    print(f"\n=== Heatmap Summary ===")
    print(f"  thigh_length = {best_thigh_length:.4f} m")
    print(f"  leg_length   = {best_leg_length:.4f} m")
    print(f"  rho          = {RHO_FIXED:.4f} kg/m (fixed)")
    print(f"  Best reward  = {best_reward:.4f}")

    np.savez(output_path.replace(".png", "_data.npz"),
             thigh_lengths=thigh_lengths, leg_lengths=leg_lengths,
             heatmap=heatmap, rho_fixed=RHO_FIXED)
    return heatmap, thigh_lengths, leg_lengths


if __name__ == "__main__":
    theta = np.array([0.258,0.635,1.932])
    debug_controller(theta)