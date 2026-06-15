import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import json
import jax
import jax.numpy as jnp

from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive
from hopper_codesign.tasks.hopper_task import HopperLocomotionTask
from hopper_codesign.assets.model_builder import build_hopper_model


def plot_convergence(history_path: str = "results/results_hopper/hopper_optimization_log.json"):
    """Plot best fitness and design parameters over generations."""

    default_history = {
        "generations": [0],
        "best_fitness": [0.0],
        "best_theta": [[0.35, 0.4, 2.0]]  # [thigh_length, leg_length, rho]
    }

    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    if not os.path.exists(history_path) or os.path.getsize(history_path) == 0:
        print(f"No valid history found. Creating default file at '{history_path}'...")
        with open(history_path, 'w') as f:
            json.dump(default_history, f, indent=2)
        history = default_history
    else:
        with open(history_path, 'r') as f:
            history = json.load(f)

    generations  = history['generations']
    best_fitness = history['best_fitness']
    best_thetas  = np.array(history['best_theta'])

    # 3 params + 1 fitness = 4 plots → 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Hopper Co-Design: CMA-ES Convergence", fontsize=14)

    # Fitness over generations
    axes[0, 0].plot(generations, best_fitness, 'b-o', linewidth=2)
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].set_ylabel("Best Reward")
    axes[0, 0].set_title("Best Design Performance per Generation")
    axes[0, 0].grid(True)

    # Design parameter evolution — 3 params
    param_names = [
        'Thigh Length (m)',
        'Leg Length (m)',
        'Density rho (kg/m)',
    ]
    colors = ['r', 'g', 'b']
    plot_positions = [(0, 1), (1, 0), (1, 1)]

    for i, (name, color, pos) in enumerate(zip(param_names, colors, plot_positions)):
        ax = axes[pos]
        ax.plot(generations, best_thetas[:, i], f'{color}-o', linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel(name)
        ax.set_title(f"Optimal {name} over Generations")
        ax.grid(True)

    plt.tight_layout()

    out_path = "results/results_hopper/hopper_convergence.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved convergence plot to {out_path}")


def render_best_design(theta: np.ndarray):
    print("Step 1: building model...")
    mj_model = build_hopper_model(theta)
    print("Step 2: creating MjData...")
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    thigh_length, leg_length = theta[0], theta[1]
    rest_height = 0.5 + thigh_length + leg_length
    print(f"Rest height: {rest_height:.3f}m")

    print("\n=== qpos at start ===", mj_data.qpos)
    print("=== qvel at start ===", mj_data.qvel)
    print("=== nq nv nu ===", mj_model.nq, mj_model.nv, mj_model.nu)

    print("Step 3: creating task...")
    task = HopperLocomotionTask(
        mj_model,
        rest_height=rest_height,
        healthy_angle_range=(-0.4, 0.4),
    )
    print("Step 4: creating MPPI controller...")
    controller = MPPI(task, num_samples=256, noise_level=0.5, temperature=0.1, plan_horizon=1.0, num_knots=5)

    run_interactive(
        controller=controller,
        mj_model=mj_model,
        mj_data=mj_data,
        frequency=60.0,
    )
    print("Step 6: viewer closed.")