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
PLAN_HORIZON = 0.5
NUM_KNOTS = 4
NUM_SAMPLES = 1024
NOISE_LEVEL = 0.3
TEMPERATURE = 0.5
EPISODE_STEPS = 300
NUM_EPISODES = 2
BETA_OPT_ITER = 1.0
BETA_HORIZON = 1.0
TEMPERATURE = 0.001


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


if __name__ == "__main__":
    theta = np.array([0.35, 0.4, 2.0])
    debug_controller(theta)