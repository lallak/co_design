import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from matplotlib import pyplot as plt
from mujoco import mjx
from hydrax.algs import DIAL
from functools import lru_cache

from humanoid_codesign.tasks.humanoid_task import HumanoidLocomotionTask
from humanoid_codesign.assets.model_builder import build_humanoid_model, get_rest_height

# Hyperparameters
PLAN_HORIZON  = 1.0
NUM_KNOTS     = 4
NUM_SAMPLES   = 1024
NOISE_LEVEL   = 0.3
TEMPERATURE   = 0.5
EPISODE_STEPS = 300
NUM_EPISODES  = 2
BETA_OPT_ITER = 1.0
BETA_HORIZON  = 1.0

JOINT_NAMES = [
    "L_hip_roll", "L_hip_yaw", "L_hip_pitch", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_roll", "R_hip_yaw", "R_hip_pitch", "R_knee", "R_ankle_pitch", "R_ankle_roll",
]


@lru_cache(maxsize=256)
def _build_controller(theta_tuple):
    theta = np.array(theta_tuple)
    mj_model  = build_humanoid_model(theta)
    mjx_model = mjx.put_model(mj_model)
    task      = HumanoidLocomotionTask(mj_model)

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
    jit_optimize   = jax.jit(controller.optimize)
    jit_get_action = jax.jit(controller.get_action)
    return mj_model, mjx_model, task, controller, jit_optimize, jit_get_action


def evaluate_design(theta: np.ndarray, seed: int = 0) -> float:
    theta_key = tuple(np.round(theta, 4))
    mj_model, mjx_model, task, controller, jit_optimize, jit_get_action = _build_controller(theta_key)

    mj_data_init = mujoco.MjData(mj_model)
    mj_data_init.qpos[2] = get_rest_height(theta)  # z height (freejoint)
    mj_data_init.qpos[3] = 1.0                      # quaternion w, upright
    mj_data_init.qvel[:] = 0.0

    # Slight crouched standing posture with wider lateral stance.
    stance_hip_roll = 0.14
    mj_data_init.qpos[7] = stance_hip_roll     # Left hip roll
    mj_data_init.qpos[9] = -0.20  # Left hip pitch
    mj_data_init.qpos[10] = 0.40  # Left knee
    mj_data_init.qpos[11] = -0.20  # Left ankle pitch

    mj_data_init.qpos[13] = -stance_hip_roll   # Right hip roll
    mj_data_init.qpos[15] = -0.20  # Right hip pitch
    mj_data_init.qpos[16] = 0.40  # Right knee
    mj_data_init.qpos[17] = -0.20  # Right ankle pitch

    def run_single_episode(episode_seed_key):
        perturb_key, _ = jax.random.split(episode_seed_key)
        perturb = jax.random.uniform(perturb_key, shape=(), minval=-0.02, maxval=0.02)

        mjx_data = mjx.put_data(mj_model, mj_data_init)
        mjx_data = mjx_data.replace(qpos=mjx_data.qpos.at[2].add(perturb))

        initial_knots = jnp.zeros((NUM_KNOTS, mj_model.nu), dtype=jnp.float32)
        ctrl_state    = controller.init_params(initial_knots=initial_knots, seed=0)

        def step_fn(carry, _):
            mjx_data, ctrl_state = carry
            ctrl_state, _ = jit_optimize(mjx_data, ctrl_state)
            action   = jit_get_action(ctrl_state, mjx_data.time)
            mjx_data = mjx_data.replace(ctrl=action)
            mjx_data = mjx.step(mjx_model, mjx_data)
            cost     = task.running_cost(mjx_data, action)
            return (mjx_data, ctrl_state), cost

        (final_data, _), costs = jax.lax.scan(
            step_fn, (mjx_data, ctrl_state), None, length=EPISODE_STEPS
        )

        return final_data.qpos[0] - jnp.sum(costs)

    rng          = jax.random.PRNGKey(seed)
    episode_keys = jax.random.split(rng, NUM_EPISODES)
    rewards      = jax.jit(jax.vmap(run_single_episode))(episode_keys)
    return float(jnp.mean(rewards))


def debug_controller(theta: np.ndarray):
    theta_key = tuple(np.round(theta, 4))
    mj_model, mjx_model, task, controller, jit_optimize, jit_get_action = _build_controller(theta_key)
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[2] = get_rest_height(theta)
    mj_data.qpos[3] = 1.0
    mj_data.qvel[:] = 0.0

    # Slight crouched standing posture with wider lateral stance.
    stance_hip_roll = 0.14
    mj_data.qpos[7] = stance_hip_roll     # Left hip roll
    mj_data.qpos[9] = -0.20  # Left hip pitch
    mj_data.qpos[10] = 0.40  # Left knee
    mj_data.qpos[11] = -0.20  # Left ankle pitch

    mj_data.qpos[13] = -stance_hip_roll   # Right hip roll
    mj_data.qpos[15] = -0.20  # Right hip pitch
    mj_data.qpos[16] = 0.40  # Right knee
    mj_data.qpos[17] = -0.20  # Right ankle pitch

    mjx_data      = mjx.put_data(mj_model, mj_data)
    initial_knots = jnp.zeros((NUM_KNOTS, mj_model.nu), dtype=jnp.float32)
    ctrl_state    = controller.init_params(initial_knots=initial_knots, seed=0)

    heights, forward_vels, rewards = [], [], []
    actions_log = []

    for _ in range(EPISODE_STEPS):
        ctrl_state, _ = jit_optimize(mjx_data, ctrl_state)
        action   = jit_get_action(ctrl_state, mjx_data.time)
        mjx_data = mjx_data.replace(ctrl=action)
        mjx_data = mjx.step(mjx_model, mjx_data)

        heights.append(float(mjx_data.qpos[2]))
        forward_vels.append(float(mjx_data.qvel[0]))
        rewards.append(float(-task.running_cost(mjx_data, action)))
        actions_log.append(np.array(action))

    actions_log = np.array(actions_log)

    print(f"Final x distance : {float(mjx_data.qpos[0]):.3f} m")
    print(f"Final z height   : {float(mjx_data.qpos[2]):.3f} m")

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(heights);      axs[0].set_title("Torso Height (z)"); axs[0].grid(True)
    axs[1].plot(forward_vels); axs[1].set_title("Forward Velocity"); axs[1].grid(True)
    axs[2].plot(rewards);      axs[2].set_title("Reward");           axs[2].grid(True)
    plt.tight_layout(); plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for i in range(6):
        axs[0].plot(actions_log[:, i],   label=JOINT_NAMES[i])
        axs[1].plot(actions_log[:, i+6], label=JOINT_NAMES[i+6])
    for ax, side in zip(axs, ["Left leg", "Right leg"]):
        ax.set_title(f"Motor Commands — {side}")
        ax.set_xlabel("Step"); ax.set_ylabel("Torque")
        ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.show()


def sensitivity_analysis(nominal_theta: np.ndarray, n_eval: int = 5, delta: float = 0.1) -> dict:
    param_names = ["thigh_length", "shank_length", "rho"]

    base_reward = evaluate_design(nominal_theta, seed=0)
    print(f"Base reward: {base_reward:.2f}")

    sensitivities = {}
    for i, name in enumerate(param_names):
        perturbed    = nominal_theta.copy()
        perturbed[i] *= (1 + delta)
        perturbed_reward = np.mean([evaluate_design(perturbed, seed=s) for s in range(n_eval)])
        sensitivities[name] = abs(perturbed_reward - base_reward)
        print(f"  {name:>15}: Δreward = {sensitivities[name]:.3f}")

    return sensitivities


if __name__ == "__main__":
    theta = np.array([0.15, 0.16, 2.0])
    debug_controller(theta)
