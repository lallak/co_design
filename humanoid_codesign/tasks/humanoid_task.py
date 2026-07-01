import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.task_base import Task


class HumanoidLocomotionTask(Task):

    def __init__(
            self,
            mj_model: mujoco.MjModel,
            rest_height: float = 0.85,
            forward_reward_weight: float = 0.5,
            ctrl_cost_weight: float = 1e-3,
            healthy_reward: float = 2.0,
            healthy_z_min: float = 0.65,
    ):
        super().__init__(mj_model)
        self.rest_height           = rest_height
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight      = ctrl_cost_weight
        self.healthy_reward        = healthy_reward
        self.healthy_z_min         = healthy_z_min

    def _is_healthy(self, x: mjx.Data) -> jax.Array:
        height = x.qpos[2]
        qw, qx, qy, qz = x.qpos[3], x.qpos[4], x.qpos[5], x.qpos[6]

        # Pitch: forward/backward tilt
        pitch = jnp.arcsin(2.0 * (qw * qy - qz * qx))
        # Roll: lateral tilt — critical for a biped with no upper body to balance
        roll  = jnp.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))

        height_ok = height >= self.healthy_z_min
        pitch_ok  = jnp.abs(pitch) <= 0.4  # ~23 deg
        roll_ok   = jnp.abs(roll)  <= 0.3  # ~17 deg, tighter: no arms to recover
        return height_ok & pitch_ok & roll_ok

    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:
        height = x.qpos[2]
        qw, qx, qy, qz = x.qpos[3], x.qpos[4], x.qpos[5], x.qpos[6]
        pitch = jnp.arcsin(2.0 * (qw * qy - qz * qx))
        roll  = jnp.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))

        forward_reward = self.forward_reward_weight * x.qvel[0]
        ctrl_cost      = self.ctrl_cost_weight * jnp.sum(jnp.square(u))

        is_healthy    = self._is_healthy(x)
        healthy_bonus = jnp.where(is_healthy, self.healthy_reward, -20.0) #HIGH FALL PENALTY

        height_reward = -5.0 * jnp.minimum(height - self.rest_height,0.0) ** 2
        pitch_reward  = -4.0 * pitch ** 2
        roll_reward   = -6.0 * roll  ** 2  # weighted higher: lateral falls are unrecoverable without arms

        reward = healthy_bonus + forward_reward + height_reward + pitch_reward + roll_reward - ctrl_cost
        return -reward

    def terminal_cost(self, x: mjx.Data) -> float:
        is_healthy    = self._is_healthy(x)
        healthy_bonus = jnp.where(is_healthy, self.healthy_reward, 0.0)
        return -(self.forward_reward_weight * x.qvel[0] + healthy_bonus)