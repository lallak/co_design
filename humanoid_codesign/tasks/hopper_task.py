import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.task_base import Task


class HopperLocomotionTask(Task):
    """Hopper locomotion task for co-design, aligned with Gymnasium Hopper-v5."""

    def __init__(
            self,
            mj_model: mujoco.MjModel,
            rest_height: float = 1.25,
            forward_reward_weight: float = 1.0,
            ctrl_cost_weight: float = 1e-3,
            healthy_reward: float = 1.0,
            healthy_z_min: float = 0.7,
            healthy_angle_range: tuple = (-0.2, 0.2),
    ):
        super().__init__(mj_model)
        self.rest_height = rest_height
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_z_min = healthy_z_min
        self.healthy_angle_min = healthy_angle_range[0]
        self.healthy_angle_max = healthy_angle_range[1]

    def _is_healthy(self, x: mjx.Data) -> jax.Array:
        height = x.qpos[1]
        angle  = x.qpos[2]
        # Seuil minimum = 60% de la hauteur de repos
        height_ok = height >= (self.rest_height * 0.6)
        angle_ok  = (angle >= self.healthy_angle_min) & (angle <= self.healthy_angle_max)
        return jnp.logical_and(height_ok, angle_ok)

    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:
        height = x.qpos[1]
        angle = x.qpos[2]

        forward_velocity = x.qvel[0]
        forward_reward = self.forward_reward_weight * forward_velocity
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(u))

        is_healthy = self._is_healthy(x)
        healthy_bonus = jnp.where(is_healthy, self.healthy_reward, -5.0)

        angle_reward = -2.0 * angle ** 2
        target_height = self.rest_height
        height_reward = -2.0 * (height - target_height) ** 2

        reward = (
                healthy_bonus
                + forward_reward
                + angle_reward
                + height_reward
                - ctrl_cost
        )
        return -reward

    def terminal_cost(self, x: mjx.Data) -> float:
        """
        Terminal cost: penalize low forward velocity and unhealthy final state.
        """
        forward_velocity = x.qvel[0]
        forward_reward = self.forward_reward_weight * forward_velocity

        is_healthy = self._is_healthy(x)
        healthy_bonus = jnp.where(is_healthy, self.healthy_reward, 0.0)

        return -(forward_reward + healthy_bonus)