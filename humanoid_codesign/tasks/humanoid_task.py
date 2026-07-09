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
            target_speed: float = 0.2,
            forward_reward_weight: float = 5.0,
            posture_cost_weight: float = 0.005,
            ctrl_cost_weight: float = 1e-3,
            healthy_reward: float = 0.5,
            healthy_z_min: float = 0.65,
    ):
        super().__init__(mj_model)
        self.rest_height           = rest_height
        self.target_speed = target_speed
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.posture_cost_weight = posture_cost_weight
        self.healthy_reward        = healthy_reward
        self.healthy_z_min         = healthy_z_min

        self.left_foot_body = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "leg_left_ankle_roll",
        )

        self.right_foot_body = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "leg_right_ankle_roll",
        )

        self.qstand = jnp.array([
            # Left leg
            0.0,  # hip roll
            0.0,  # hip yaw
            -0.20,  # hip pitch
            0.40,  # knee
            -0.20,  # ankle pitch
            0.0,  # ankle roll

            # Right leg
            0.0,  # hip roll
            0.0,  # hip yaw
            -0.20,  # hip pitch
            0.40,  # knee
            -0.20,  # ankle pitch
            0.0,  # ankle roll
        ], dtype=jnp.float32)

    def _is_healthy(self, x: mjx.Data) -> jax.Array:
        height = x.qpos[2]
        qw, qx, qy, qz = x.qpos[3], x.qpos[4], x.qpos[5], x.qpos[6]

        # Pitch: forward/backward tilt
        '''pitch = jnp.arcsin(2.0 * (qw * qy - qz * qx))
        # Roll: lateral tilt — critical for a biped with no upper body to balance
        roll  = jnp.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))
'''
        height_ok = height >= self.healthy_z_min
        '''pitch_ok  = jnp.abs(pitch) <= 0.4  # ~23 deg
        roll_ok   = jnp.abs(roll)  <= 0.3  # ~17 deg, tighter: no arms to recover'''
        return height_ok

    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:
        height = x.qpos[2]
        qw, qx, qy, qz = x.qpos[3], x.qpos[4], x.qpos[5], x.qpos[6]
        pitch = jnp.arcsin(2.0 * (qw * qy - qz * qx))
        roll  = jnp.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))

        #reward following speed
        forward_reward = self.forward_reward_weight * x.qvel[0]
        #forward_reward = -(x.qvel[0] - self.target_speed) ** 2

        #penalize large actuator torques, more efficient motion
        ctrl_cost      = self.ctrl_cost_weight * jnp.sum(jnp.square(u))

        #staying upright
        is_healthy    = self._is_healthy(x)
        #healthy_reward = jnp.where(is_healthy, self.healthy_reward, -20.0) #HIGH FALL PENALTY
        healthy_reward = jnp.where(is_healthy, 1.0, -1.0)

        #remain near standing height
        height_reward = -5.0 * jnp.minimum(height - self.rest_height,0.0) ** 2
        #penalize leaning forward or backward
        pitch_reward  = -1.0 * pitch ** 2
        #penalize leaning on sides
        roll_reward   = -2.0 * roll  ** 2  # weighted higher: lateral falls are unrecoverable without arms

        # Encourage a slight forward lean (~0.15 rad ≈ 9°)
        desired_pitch = 0.15
        torso_forward_reward = 0.5 * jnp.exp(-20.0 * (pitch - desired_pitch) ** 2)

        '''posture_cost = (
                self.posture_cost_weight
                * jnp.sum(
            (x.qpos[7:] - self.qstand) ** 2
        )
        )'''

        #vertical_velocity_cost = 0.5 * x.qvel[2] ** 2

        left_pos = x.xpos[self.left_foot_body]
        right_pos = x.xpos[self.right_foot_body]

        #reward only when one foot higher than the other and in front
        foot_diff_reward = jnp.abs(left_pos[2] - right_pos[2])
        foot_forward_reward = jnp.abs(left_pos[0] - right_pos[0])

        #keep the robot going straight
        lateral_cost = 2.0 * x.qvel[1] ** 2
        yaw_rate_cost = 0.5 * x.qvel[5] ** 2


        reward = healthy_reward + forward_reward + height_reward + foot_diff_reward + foot_forward_reward + pitch_reward + roll_reward + torso_forward_reward - ctrl_cost - lateral_cost - yaw_rate_cost

        return -reward

    def terminal_cost(self, x: mjx.Data) -> float:
        is_healthy    = self._is_healthy(x)
        healthy_reward = jnp.where(is_healthy, self.healthy_reward, -2.0)
        return -healthy_reward