import sys
from pathlib import Path

# Add hydrax directory to path so hydrax imports work
hydrax_parent = Path(__file__).parent / "hydrax"
sys.path.insert(0, str(hydrax_parent))

import numpy as np
import matplotlib.pyplot as plt
import json
import mujoco
from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive

from cartpole_codesign.tasks.cartpole_task import CartpoleSwingupTask
from cartpole_codesign.assets.model_builder import build_cartpole_model


def plot_convergence(history_path: str = "results_cartpole/optimization_log.json"):
    """Plot best fitness and design parameters over generations."""

    with open(history_path, 'r') as f:
        history = json.load(f)

    generations = history['generations']
    best_fitness = history['best_fitness']
    best_thetas = np.array(history['best_theta'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("CartPole Co-Design: CMA-ES Convergence", fontsize=14)

    # Fitness over generations
    axes[0, 0].plot(generations, best_fitness, 'b-o', linewidth=2)
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].set_ylabel("Best Reward")
    axes[0, 0].set_title("Best Design Performance per Generation")
    axes[0, 0].grid(True)

    # Design parameter evolution
    #param_names = ['Pole Length (m)', 'Pole Mass (kg)', 'Cart Mass (kg)']
    param_names = ['Pole Length (m)', 'Cart Mass (kg)']
    colors = ['r', 'g', 'b']

    for i, (name, color) in enumerate(zip(param_names, colors)):
        ax = axes[(i + 1) // 2, (i + 1) % 2]
        ax.plot(generations, best_thetas[:, i], f'{color}-o', linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel(name)
        ax.set_title(f"Optimal {name} over Generations")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("results_cartpole/convergence_30_12_slowmode.png", dpi=150)
    plt.show()
    print("Saved convergence plot to results_cartpole/convergence_30_12_slowmode.png")


def render_best_design(theta: np.ndarray):
    """
    Render an interactive simulation of the best design using Hydrax viewer.

    Args:
        theta: Optimal design [pole_length, cart_mass]
               (pole_mass is fixed at 0.1 kg)
    """
    from cartpole_codesign.evaluator import POLE_MASS_FIXED

    pole_length = theta[0]
    cart_mass = theta[1]
    pole_mass = POLE_MASS_FIXED
    theta_full = np.array([pole_length, pole_mass, cart_mass])

    print(f"Rendering design: pole_length={pole_length:.3f} m, "
          f"cart_mass={cart_mass:.3f} kg, pole_mass={pole_mass:.3f} kg (fixed)")

    mj_model = build_cartpole_model(theta_full)
    task = CartpoleSwingupTask(mj_model)
    controller = MPPI(task, num_samples=512, noise_level=0.5, temperature=0.5,
                      plan_horizon=1.0, num_knots=4)

    # Initialize MuJoCo data
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[1] = np.pi  # pole hanging down initially
    mujoco.mj_forward(mj_model, mj_data)

    # Defensive check: ensure we pass a mujoco.MjModel to run_interactive.
    # Some task objects also expose an `mj_model` attribute; prefer that if
    # caller accidentally passed a Task instead of an MjModel.
    if not hasattr(mj_model, "opt"):
        # try to recover from a task-like object
        if hasattr(task, "mj_model"):
            real_mj_model = task.mj_model
            print("Note: using task.mj_model for simulation (recovered).")
        else:
            raise TypeError("mj_model does not appear to be a MuJoCo MjModel and task has no mj_model")
    else:
        real_mj_model = mj_model

    frequency = 100  # Hz
    print(f"Starting interactive run at {frequency} Hz using model type: {type(real_mj_model)}")
    run_interactive(controller, real_mj_model, mj_data, frequency)



