import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import json

from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive
from humanoid_codesign.tasks.humanoid_task import HumanoidLocomotionTask
from humanoid_codesign.assets.model_builder import build_humanoid_model, get_rest_height


def plot_convergence(history_path: str = "results/results_humanoid/humanoid_optimization_log.json"):
    """Plot best fitness and design parameters over generations."""

    default_history = {
        "generations": [0],
        "best_fitness": [0.0],
        "best_theta": [[0.15, 0.16, 2.0]]  # [thigh_length, shank_length, rho]
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Humanoid Co-Design: CMA-ES Convergence", fontsize=14)

    axes[0, 0].plot(generations, best_fitness, 'b-o', linewidth=2)
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].set_ylabel("Best Reward")
    axes[0, 0].set_title("Best Design Performance per Generation")
    axes[0, 0].grid(True)

    param_names = [
        'Thigh Length (m)',
        'Shank Length (m)',
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

    out_path = "results/results_humanoid/humanoid_convergence.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved convergence plot to {out_path}")


def render_best_design(theta: np.ndarray):
    print("Step 1: building model...")
    mj_model = build_humanoid_model(theta)
    print("Step 2: creating MjData...")
    mj_data = mujoco.MjData(mj_model)

    # No keyframe defined for this robot — set the freejoint manually instead
    rest_height = get_rest_height(theta)
    mj_data.qpos[2] = rest_height  # z height
    mj_data.qpos[3] = 1.0          # quaternion w, upright
    mj_data.qvel[:] = 0.0
    # Standing posture
    mj_data.qpos[7] = 0.08  # L hip roll
    mj_data.qpos[9] = -0.20  # L hip pitch
    mj_data.qpos[10] = 0.40  # L knee
    mj_data.qpos[11] = -0.20  # L ankle pitch

    mj_data.qpos[13] = -0.08  # R hip roll
    mj_data.qpos[15] = -0.20  # R hip pitch
    mj_data.qpos[16] = 0.40  # R knee
    mj_data.qpos[17] = -0.20  # R ankle pitch
    mujoco.mj_forward(mj_model, mj_data)

    print(f"Rest height: {rest_height:.3f}m")

    print("\n=== qpos at start ===", mj_data.qpos)
    print("=== qvel at start ===", mj_data.qvel)
    print("=== nq nv nu ===", mj_model.nq, mj_model.nv, mj_model.nu)

    print("Step 3: creating task...")
    task = HumanoidLocomotionTask(
        mj_model,
        rest_height=rest_height,
        healthy_z_min=rest_height * 0.7,
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