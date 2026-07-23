import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.task_base import Task


class HumanoidLocomotionTask(Task):

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        rest_height: float = 0.85,  # 0.85
        healthy_z_min: float = 0.65,

        # -------- Commanded walking --------
        target_vx: float = 0.3,
        target_vy: float = 0.0,
        target_yaw_rate: float = 0.0,

        # -------- Reward weights --------
        gait_weight: float = 1.0,
        velocity_weight: float = 6.0,
        angular_velocity_weight: float = 0.3,
        upright_weight: float = 3.0,
        yaw_weight: float = 0.2,
        height_weight: float = 3.0,
        energy_weight: float = 0.02,
        air_time_weight: float = 0.0,
        weight_shift_weight: float = 1.5,
        swing_pose_weight: float = 0.5,
        step_weight: float = 2.0,
        swing_forward_weight: float = 1.5,
        posture_weight: float = 0.2,
        alive_weight: float = 2.0,
        ref_tracking_weight: float = 8.0,

        # -------- Gait parameters --------
        gait: str = "walk",
    ):
        super().__init__(mj_model)

        self.rest_height = rest_height
        self.healthy_z_min = healthy_z_min

        self.target_vx = target_vx
        self.target_vy = target_vy
        self.target_yaw_rate = target_yaw_rate

        self.gait_weight = gait_weight
        self.velocity_weight = velocity_weight
        self.angular_velocity_weight = angular_velocity_weight
        self.upright_weight = upright_weight
        self.yaw_weight = yaw_weight
        self.height_weight = height_weight
        self.energy_weight = energy_weight
        self.air_time_weight = air_time_weight
        self.weight_shift_weight = weight_shift_weight
        self.swing_pose_weight = swing_pose_weight
        self.step_weight = step_weight
        self.swing_forward_weight = swing_forward_weight
        self.posture_weight = posture_weight
        self.alive_weight = alive_weight
        self.ref_tracking_weight = ref_tracking_weight

        self.gait = gait

        # -------------------------------------------------
        # Feet
        # -------------------------------------------------

        self.left_foot_body = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "leg_left_ankle_roll",
        )
        self.right_foot_body = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "leg_right_ankle_roll",
        )
        self.feet = jnp.array(
            [self.left_foot_body, self.right_foot_body], dtype=jnp.int32,
        )

        # -------------------------------------------------
        # Gait parameters (Unitree-style phase clock)
        # phase offset (left, right) as a fraction of the full gait cycle
        # -------------------------------------------------

        self.gait_phase = {
            "stand": jnp.array([0.0, 0.0]),
            "slow_walk": jnp.array([0.0, 0.5]),
            "walk": jnp.array([0.0, 0.5]),
            "jog": jnp.array([0.0, 0.5]),
        }

        # duty ratio, cadence (full gait cycles per second), swing height (m)
        self.gait_params = {
            "stand": jnp.array([1.0, 1.0, 0.0]),
            "slow_walk": jnp.array([0.6, 0.8, 0.15]),
            "walk": jnp.array([0.5, 1.0, 0.15]),
            "jog": jnp.array([0.3, 2.0, 0.20]),
        }

        # -------------------------------------------------
        # Standing pose
        # -------------------------------------------------

        self.qstand = jnp.array([
            0.0, 0.0, -0.08, 0.15, -0.08, 0.0,
            0.0, 0.0, -0.08, 0.15, -0.08, 0.0,
        ])

    def _gait_signals(self, t):
        """
        Time-based gait clock. Returns, for each foot:
          - stance (1.0 in stance / 0.0 in swing), as a smooth-ish 0/1 float
          - target swing height trajectory (0 in stance, half-sine bump in swing)
        This replaces the old absolute-z contact threshold, which is fragile
        because it depends on exactly where the ankle body origin sits above
        the ground (foot sole thickness, geometry offsets, etc.) and can end
        up permanently "no contact" if that offset is off, which breaks every
        reward term that depends on it.
        """
        duty, cadence, swing_height = (
            self.gait_params[self.gait][0],
            self.gait_params[self.gait][1],
            self.gait_params[self.gait][2],
        )
        offset_left, offset_right = (
            self.gait_phase[self.gait][0],
            self.gait_phase[self.gait][1],
        )

        phase_left = jnp.mod(t * cadence + offset_left, 1.0)
        phase_right = jnp.mod(t * cadence + offset_right, 1.0)

        stance_left = (phase_left < duty).astype(jnp.float32)
        stance_right = (phase_right < duty).astype(jnp.float32)

        # normalized progress through the swing portion of the cycle, in [0, 1]
        swing_frac_left = jnp.clip(
            (phase_left - duty) / jnp.maximum(1.0 - duty, 1e-6), 0.0, 1.0
        )
        swing_frac_right = jnp.clip(
            (phase_right - duty) / jnp.maximum(1.0 - duty, 1e-6), 0.0, 1.0
        )

        target_z_left = (1.0 - stance_left) * swing_height * jnp.sin(jnp.pi * swing_frac_left)
        target_z_right = (1.0 - stance_right) * swing_height * jnp.sin(jnp.pi * swing_frac_right)

        return stance_left, stance_right, target_z_left, target_z_right, swing_frac_left, swing_frac_right

    def _reference_leg_trajectory(self, t):
        """
        Explicit joint-space reference trajectory for the 12 leg DOF, driven
        by the same phase clock as the gait signals. This gives the sampler
        something to TRACK rather than something to discover from scratch --
        sampling-based MPC (PS/MPPI/CEM) is known to get stuck in a static
        local minimum (e.g. a stable squat) when it has to invent a full
        coordinated stepping motion via random perturbation. Tracking a
        hand-authored nominal gait is a far easier landscape to optimize.

        Layout per leg (matches qpos[7:13] / qpos[13:19]):
          [hip_roll, hip_yaw, hip_pitch, knee, ankle_pitch, ankle_roll]

        This is a first-pass, deliberately simple reference -- tune the
        amplitudes below to your robot's actual leg geometry/limits.
        """
        stance_left, stance_right, target_z_left, target_z_right, swing_frac_left, swing_frac_right = self._gait_signals(t)

        hip_pitch_amp = 0.35   # rad, how far the hip swings fore/aft
        knee_swing_amp = 0.55  # rad, extra knee bend during swing

        base_hip = self.qstand[2]
        base_knee = self.qstand[3]
        base_ankle = self.qstand[4]

        # Hip pitch ramps from trailing (+amp) to leading (-amp) over the
        # course of the swing phase; during stance it ramps the other way
        # (mirrors the opposite leg's swing) so the body rolls forward over it.
        hip_pitch_left = base_hip + hip_pitch_amp * jnp.where(
            stance_left > 0.5,
            jnp.cos(jnp.pi * swing_frac_right),   # stance: mirror the other leg
            -jnp.cos(jnp.pi * swing_frac_left),
        )
        hip_pitch_right = base_hip + hip_pitch_amp * jnp.where(
            stance_right > 0.5,
            jnp.cos(jnp.pi * swing_frac_left),
            -jnp.cos(jnp.pi * swing_frac_right),
        )

        # Knee bends extra during swing (foot clearance), stays near
        # standing bend during stance.
        knee_left = base_knee + knee_swing_amp * (1.0 - stance_left) * jnp.sin(jnp.pi * swing_frac_left)
        knee_right = base_knee + knee_swing_amp * (1.0 - stance_right) * jnp.sin(jnp.pi * swing_frac_right)

        # Ankle roughly compensates hip+knee to keep the foot flat-ish;
        # kept simple/constant here as a starting point.
        ankle_pitch_left = base_ankle
        ankle_pitch_right = base_ankle

        q_ref = jnp.array([
            0.0, 0.0, hip_pitch_left, knee_left, ankle_pitch_left, 0.0,
            0.0, 0.0, hip_pitch_right, knee_right, ankle_pitch_right, 0.0,
        ])
        return q_ref

    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:
        # ==========================================================
        # Base state
        # ==========================================================

        height = x.qpos[2]
        qw, qx, qy, qz = x.qpos[3:7]

        sin_pitch = jnp.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = jnp.arcsin(sin_pitch)
        roll = jnp.arctan2(
            2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy),
        )
        yaw = jnp.arctan2(
            2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz),
        )

        # ==========================================================
        # Velocity tracking (now uses the *configured* targets, not
        # hardcoded ones — previously this ignored self.target_vx/vy/yaw
        # and always chased 1.0 m/s regardless of what was passed in,
        # which also disagreed with terminal_cost's target)
        # ==========================================================

        vx = x.qvel[0]
        vy = x.qvel[1]

        reward_vel = -((vx - self.target_vx) ** 2 + (vy - self.target_vy) ** 2)

        roll_rate = x.qvel[3]
        pitch_rate = x.qvel[4]
        yaw_rate = x.qvel[5]

        reward_yaw_rate = -(yaw_rate - self.target_yaw_rate) ** 2
        reward_ang_vel = -(roll_rate ** 2 + pitch_rate ** 2 + yaw_rate ** 2)

        # ==========================================================
        # Upright / yaw / height
        # ==========================================================

        reward_upright = -(pitch ** 2 + roll ** 2)
        reward_yaw = -(yaw ** 2)
        reward_height = -(height - self.rest_height) ** 2

        # ==========================================================
        # Standing posture reward
        # ==========================================================

        qpos_joints = x.qpos[7:19]
        reward_posture = -jnp.sum((qpos_joints - self.qstand) ** 2)

        # ==========================================================
        # Phase-based gait clock (replaces the old absolute-z contact
        # estimate, which was almost certainly always reading "no
        # contact" for both feet — see explanation above)
        # ==========================================================

        stance_left, stance_right, target_z_left, target_z_right, _, _ = self._gait_signals(x.time)
        swing_left = 1.0 - stance_left
        swing_right = 1.0 - stance_right

        # -- reference trajectory tracking: this is the primary driver of
        #    the stepping motion. A static crouch is a stable local minimum
        #    for sampling-based MPC because discovering a coordinated swing
        #    from scratch via random perturbation is hard; tracking a known
        #    walking reference turns that into a much easier "stay close to
        #    this nominal" problem. --
        q_ref = self._reference_leg_trajectory(x.time)
        reward_ref_tracking = -jnp.sum((x.qpos[7:19] - q_ref) ** 2)

        left_z = x.xpos[self.left_foot_body][2]
        right_z = x.xpos[self.right_foot_body][2]

        pelvis_y = x.qpos[1]
        left_y = x.xpos[self.left_foot_body][1]
        right_y = x.xpos[self.right_foot_body][1]

        left_x = x.xpos[self.left_foot_body][0]
        right_x = x.xpos[self.right_foot_body][0]

        left_vx = x.cvel[self.left_foot_body][3]
        right_vx = x.cvel[self.right_foot_body][3]

        left_hip_pitch = x.qpos[9]
        left_knee = x.qpos[10]
        right_hip_pitch = x.qpos[15]
        right_knee = x.qpos[16]

        # -- weight shift: pelvis over whichever foot is in stance --
        reward_weight_shift = (
            stance_left * swing_right * (-(pelvis_y - left_y) ** 2)
            + stance_right * swing_left * (-(pelvis_y - right_y) ** 2)
        )

        # -- swing leg configuration: bent knee / forward hip, but only
        #    for the leg that is actually scheduled to be swinging --
        reward_swing_pose = (
            swing_left * (-(left_hip_pitch + 1.0) ** 2 - (left_knee - 0.80) ** 2)
            + swing_right * (-(right_hip_pitch + 1.0) ** 2 - (right_knee - 0.80) ** 2)
        )

        # -- gait: track the commanded swing-height trajectory while
        #    swinging, stay low while in stance --
        reward_gait = (
            swing_left * (-(left_z - target_z_left) ** 2)
            + stance_left * (-(left_z) ** 2)
            + swing_right * (-(right_z - target_z_right) ** 2)
            + stance_right * (-(right_z) ** 2)
        )

        # -- swing-foot forward velocity while swinging --
        reward_swing_forward = (
            swing_left * jnp.maximum(left_vx, 0.0)
            + swing_right * jnp.maximum(right_vx, 0.0)
        )

        # -- step placement: swing foot ahead of the stance foot --
        left_step = swing_left * stance_right * jnp.maximum(left_x - right_x, 0.0)
        right_step = swing_right * stance_left * jnp.maximum(right_x - left_x, 0.0)
        reward_step = left_step + right_step

        # -- time spent with exactly one foot down (discourages double
        #    support / squatting from lingering) --
        reward_air_time = stance_left * swing_right + stance_right * swing_left

        # ==========================================================
        # Energy + alive
        # ==========================================================

        reward_energy = -jnp.sum(u ** 2)

        healthy = (
            (height > self.healthy_z_min)
            & (jnp.abs(pitch) < 0.6)
            & (jnp.abs(roll) < 0.5)
        )
        reward_alive = jnp.where(healthy, 1.0, -5.0)

        # ==========================================================
        # Final reward — now actually driven by the constructor's
        # weight attributes, so passing different weights in __init__
        # actually changes behavior
        # ==========================================================

        reward = (
            self.velocity_weight * reward_vel
            + self.step_weight * reward_step
            + self.swing_forward_weight * reward_swing_forward
            + self.gait_weight * reward_gait
            + self.weight_shift_weight * reward_weight_shift
            + self.angular_velocity_weight * reward_ang_vel
            + self.height_weight * reward_height
            + self.swing_pose_weight * reward_swing_pose
            + self.posture_weight * reward_posture
            + self.upright_weight * reward_upright
            + self.yaw_weight * reward_yaw
            + self.energy_weight * reward_energy
            + self.alive_weight * reward_alive
            + self.air_time_weight * reward_air_time
            + self.ref_tracking_weight * reward_ref_tracking
        )

        return -reward

    def terminal_cost(self, x: mjx.Data) -> float:
        height = x.qpos[2]
        qw, qx, qy, qz = x.qpos[3:7]

        sin_pitch = jnp.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = jnp.arcsin(sin_pitch)
        roll = jnp.arctan2(
            2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy),
        )
        yaw = jnp.arctan2(
            2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz),
        )

        vx = x.qvel[0]
        vy = x.qvel[1]
        roll_rate = x.qvel[3]
        pitch_rate = x.qvel[4]
        yaw_rate = x.qvel[5]

        terminal = 0.0
        terminal += 5.0 * (height - self.rest_height) ** 2
        terminal += 2.0 * (vx - self.target_vx) ** 2
        terminal += 2.0 * (vy - self.target_vy) ** 2
        terminal += 10.0 * pitch ** 2
        terminal += 10.0 * roll ** 2
        terminal += 2.0 * yaw ** 2
        terminal += roll_rate ** 2 + pitch_rate ** 2 + yaw_rate ** 2

        return terminal

"""import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.task_base import Task


class HumanoidLocomotionTask(Task):

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        rest_height: float = 0.85, #0.85
        healthy_z_min: float = 0.65,

        # -------- Commanded walking --------
        target_vx: float = 0.3,
        target_vy: float = 0.0,
        target_yaw_rate: float = 0.0,

        # -------- Reward weights --------
        gait_weight: float = 5.0,
        velocity_weight: float = 1.0,
        angular_velocity_weight: float = 1.0,
        upright_weight: float = 0.5,
        yaw_weight: float = 0.1,
        height_weight: float = 0.5,
        energy_weight: float = 0.01,
        air_time_weight: float = 0.0,

        # -------- Gait parameters --------
        gait: str = "walk",
    ):
        super().__init__(mj_model)

        self.rest_height = rest_height
        self.healthy_z_min = healthy_z_min

        self.target_vx = target_vx
        self.target_vy = target_vy
        self.target_yaw_rate = target_yaw_rate

        self.gait_weight = gait_weight
        self.velocity_weight = velocity_weight
        self.angular_velocity_weight = angular_velocity_weight
        self.upright_weight = upright_weight
        self.yaw_weight = yaw_weight
        self.height_weight = height_weight
        self.energy_weight = energy_weight
        self.air_time_weight = air_time_weight

        self.gait = gait

        # -------------------------------------------------
        # Feet
        # -------------------------------------------------

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

        self.feet = jnp.array(
            [self.left_foot_body, self.right_foot_body],
            dtype=jnp.int32,
        )

        # -------------------------------------------------
        # Gait parameters copied from Unitree
        # -------------------------------------------------

        self.gait_phase = {
            "stand": jnp.array([0.0, 0.0]),
            "slow_walk": jnp.array([0.0, 0.5]),
            "walk": jnp.array([0.0, 0.5]),
            "jog": jnp.array([0.0, 0.5]),
        }

        # duty ratio, cadence, swing height

        self.gait_params = {
            "stand": jnp.array([1.0, 1.0, 0.0]),
            "slow_walk": jnp.array([0.6, 0.8, 0.15]),
            "walk": jnp.array([0.5, 1.0, 0.15]),
            "jog": jnp.array([0.3, 2.0, 0.20]),
        }

        # -------------------------------------------------
        # Standing pose
        # -------------------------------------------------

        self.qstand = jnp.array([
            0.0,
            0.0,
            -0.08,
            0.15,
            - 0.08,
            0.0,

            0.0,
            0.0,
            -0.08,
            0.15,
            - 0.08,
            0.0,
        ])

    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:
        # ==========================================================
        # Desired commands
        # ==========================================================

        target_vx = 1.0  # desired forward speed
        target_vy = 0.0
        target_yaw_rate = 0.0

        # ==========================================================
        # Base state
        # ==========================================================

        height = x.qpos[2]

        qw, qx, qy, qz = x.qpos[3:7]

        sin_pitch = jnp.clip(
            2.0 * (qw * qy - qz * qx),
            -1.0,
            1.0,
        )
        pitch = jnp.arcsin(sin_pitch)

        roll = jnp.arctan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )

        yaw = jnp.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        # ==========================================================
        # Velocity tracking : reach the target velocity
        # ==========================================================

        vx = x.qvel[0]
        vy = x.qvel[1]
        yaw_rate = x.qvel[5]

        reward_vel = -(
                (vx - target_vx) ** 2
                + (vy - target_vy) ** 2
        )

        reward_yaw_rate = -(
                                   yaw_rate - target_yaw_rate
                           ) ** 2

        # ==========================================================
        # Angular velocity tracking :reduce body rotation to the min necessary
        # ==========================================================

        roll_rate = x.qvel[3]
        pitch_rate = x.qvel[4]
        yaw_rate = x.qvel[5]

        reward_ang_vel = -(
                roll_rate ** 2
                + pitch_rate ** 2
                + yaw_rate ** 2
        )

        # ==========================================================
        # Upright reward :keep torso vertical
        # ==========================================================

        reward_upright = -(pitch ** 2 + roll ** 2)

        # ==========================================================
        # Yaw reward : keep the body oriented in the desired direction
        # ==========================================================

        reward_yaw = -(yaw ** 2)

        # ==========================================================
        # Height reward:keep torso at desired height
        # ==========================================================

        reward_height = -(
                                 height - self.rest_height
                         ) ** 2

        # ==========================================================
        # Standing posture reward - reward robot to return to a nominal standing pose
        # ==========================================================

        qpos_joints = x.qpos[7:19]

        reward_posture = -jnp.sum((qpos_joints - self.qstand) ** 2)

        # ==========================================================
        # Weight-shift reward - reward pelvis to move over the stance foot
        # ==========================================================

        left_z = x.xpos[self.left_foot_body][2]
        right_z = x.xpos[self.right_foot_body][2]

        foot_contact_height = 0.04

        left_contact = jnp.clip(
            (foot_contact_height - left_z) / foot_contact_height,
            0.0,
            1.0,
        )

        right_contact = jnp.clip(
            (foot_contact_height - right_z) / foot_contact_height,
            0.0,
            1.0,
        )

        pelvis_y = x.qpos[1]

        left_y = x.xpos[self.left_foot_body][1]
        right_y = x.xpos[self.right_foot_body][1]

        left_support_reward = (
                left_contact
                * (1.0 - right_contact)
                * (-(pelvis_y - left_y) ** 2)
        )

        right_support_reward = (
                right_contact
                * (1.0 - left_contact)
                * (-(pelvis_y - right_y) ** 2)
        )

        reward_weight_shift = (
                left_support_reward
                + right_support_reward
        )

        # ==========================================================
        # Swing leg configuration - reward a bent knee and forward hip during swing
        # ==========================================================

        left_x = x.xpos[self.left_foot_body][0]
        right_x = x.xpos[self.right_foot_body][0]

        left_vx = x.cvel[self.left_foot_body][3]
        right_vx = x.cvel[self.right_foot_body][3]

        left_air = 1.0 - left_contact
        right_air = 1.0 - right_contact

        contact_reward = (
                left_contact * (1.0 - right_contact)
                + right_contact * (1.0 - left_contact)
        )

        left_hip_pitch = x.qpos[9]
        left_knee = x.qpos[10]

        right_hip_pitch = x.qpos[15]
        right_knee = x.qpos[16]

        reward_swing_pose = (
                left_air * (
                - (left_hip_pitch + 1.0) ** 2
                - (left_knee - 0.80) ** 2
        )
                +
                right_air * (
                        - (right_hip_pitch + 1.0) ** 2
                        - (right_knee - 0.80) ** 2
                )
        )

        # ==========================================================
        # Gait reward - reward the swing foot being lifted while the stance foot is on the ground
        # ==========================================================

        single_support = contact_reward

        swing_height = (
                left_contact * right_z
                + right_contact * left_z
        )

        reward_gait = (
                single_support
                * jnp.clip(swing_height / 0.15, 0.0, 1.0)
        )

        # ==========================================================
        # Swing-foot forward reward - reward the swing foot moving forward while in the air
        # ==========================================================

        reward_swing_forward = (
                left_air * jnp.maximum(left_vx, 0.0)
                + right_air * jnp.maximum(right_vx, 0.0)
        )

        # ==========================================================
        # Step placement reward - reward the swing foot being placed ahead of the stance foot
        # ==========================================================

        left_step = (
                left_air
                * right_contact
                * jnp.maximum(left_x - right_x, 0.0)
        )

        right_step = (
                right_air
                * left_contact
                * jnp.maximum(right_x - left_x, 0.0)
        )

        reward_step = left_step + right_step

        # ==========================================================
        # Energy - penalize large control inputs
        # ==========================================================

        reward_energy = -jnp.sum(u ** 2)

        # ==========================================================
        # Healthy reward :penalize falling
        # ==========================================================

        healthy = (
                (height > self.healthy_z_min)
                & (jnp.abs(pitch) < 0.6)
                & (jnp.abs(roll) < 0.5)
        )

        reward_alive = jnp.where(
            healthy,
            1.0,
            -5.0,
        )

        # ==========================================================
        # Final reward
        # ==========================================================

        reward = (
                2.0 * reward_vel #2
                + 10.0 * reward_step #10
                + 6.0 * reward_swing_forward
                + 4.0 * reward_gait
                + 10.0 * reward_weight_shift
                + 1.0 * reward_ang_vel
                + 1.5 * reward_height
                + 3.0 * reward_swing_pose
                + 2.0 * reward_posture
                + 0.0 * reward_upright
                + 0.1 * reward_yaw
                + 2.0 * reward_energy
                + 1.0 * reward_alive
        )

        return - reward


    def terminal_cost(self, x: mjx.Data) -> float:
        # ==========================================================
        # Base state
        # ==========================================================

        height = x.qpos[2]

        qw, qx, qy, qz = x.qpos[3:7]

        sin_pitch = jnp.clip(
            2.0 * (qw * qy - qz * qx),
            -1.0,
            1.0,
        )
        pitch = jnp.arcsin(sin_pitch)

        roll = jnp.arctan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )

        yaw = jnp.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        # ==========================================================
        # Base velocities
        # ==========================================================

        vx = x.qvel[0]
        vy = x.qvel[1]

        roll_rate = x.qvel[3]
        pitch_rate = x.qvel[4]
        yaw_rate = x.qvel[5]

        # ==========================================================
        # Terminal cost
        # ==========================================================

        terminal = 0.0

        # Desired height
        terminal += 5.0 * (height - self.rest_height) ** 2 #5

        # Desired forward velocity
        terminal += 2.0 * (vx - self.target_vx) ** 2
        terminal += 2.0 * (vy - self.target_vy) ** 2

        # Upright torso
        terminal += 10.0 * pitch ** 2 #10
        terminal += 10.0 * roll ** 2 #10

        # Face forward
        terminal += 2.0 * yaw ** 2 #2

        # Avoid spinning
        terminal += (
                roll_rate ** 2
                + pitch_rate ** 2
                + yaw_rate ** 2
        )

        return terminal"""


"""import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.task_base import Task


class HumanoidLocomotionTask(Task):

    def __init__(
            self,
            mj_model: mujoco.MjModel,
            rest_height: float = 0.85,
            healthy_z_min: float = 0.65,
    ):
        super().__init__(mj_model)
        self.rest_height  = rest_height
        self.healthy_z_min = healthy_z_min

        self.left_foot_body = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "leg_left_ankle_roll"
        )
        self.right_foot_body = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "leg_right_ankle_roll"
        )

    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:

        # 1. STAY UPRIGHT — most important, everything else fails if this fails
        height = x.qpos[2]
        qw, qx, qy, qz = x.qpos[3], x.qpos[4], x.qpos[5], x.qpos[6]
        pitch = jnp.arcsin(2.0 * (qw * qy - qz * qx))
        roll = jnp.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx ** 2 + qy ** 2))
        # Encourage remaining upright continuously
        pitch_balance = jnp.exp(-8.0 * pitch ** 2)
        roll_balance = jnp.exp(-12.0 * roll ** 2)

        is_healthy = (height >= self.healthy_z_min) & (jnp.abs(pitch) <= 0.6) & (jnp.abs(roll) <= 0.4)
        healthy = jnp.where(is_healthy, 1.0, -6.5)
        height_penalty = -3.0 * jnp.maximum(self.rest_height - height, 0.0) ** 2

        # 2. GO FORWARD — linear reward on forward velocity
        forward = 3.0 * jnp.maximum(x.qvel[0], 0.0)

        # 3. ALTERNATE FEET — reward one foot up while the other is down
        left_z  = x.xpos[self.left_foot_body][2]
        right_z = x.xpos[self.right_foot_body][2]
        foot_contact_height = 0.04
        left_contact  = jnp.clip((foot_contact_height - left_z)  / foot_contact_height, 0.0, 1.0)
        right_contact = jnp.clip((foot_contact_height - right_z) / foot_contact_height, 0.0, 1.0)
        # reward swing foot being lifted, but only during single support
        single_support = left_contact * (1.0 - right_contact) + right_contact * (1.0 - left_contact)
        swing_height = left_contact * right_z + right_contact * left_z  # height of the NON-contact foot
        alternation = single_support * jnp.clip(swing_height / 0.16, 0.0, 1.0)

        # reward swing foot moving forward while in the air
        left_air = 1.0 - left_contact
        right_air = 1.0 - right_contact
        swing_forward = (left_air * jnp.maximum(x.cvel[self.left_foot_body][3], 0.0) +
                         right_air * jnp.maximum(x.cvel[self.right_foot_body][3], 0.0))

        # Reward the swing foot being placed ahead of the stance foot
        left_x = x.xpos[self.left_foot_body][0]
        right_x = x.xpos[self.right_foot_body][0]

        left_step = left_air * right_contact * jnp.maximum(left_x - right_x, 0.0)
        right_step = right_air * left_contact * jnp.maximum(right_x - left_x, 0.0)

        step_reward = left_step + right_step

        # Small penalties to keep things clean
        ctrl_cost    = 1e-3 * jnp.sum(jnp.square(u))
        lateral_cost = 0.5  * x.qvel[1] ** 2

        # penalize lateral foot velocity during swing
        left_lateral = left_air * x.cvel[self.left_foot_body][4] ** 2  # y velocity of left foot
        right_lateral = right_air * x.cvel[self.right_foot_body][4] ** 2  # y velocity of right foot
        lateral_swing_cost = 2.0 * (left_lateral + right_lateral)

        # penalize feet landing too far to the side
        left_y_offset = (x.xpos[self.left_foot_body][1] - 0.08) ** 2  # 0.08 = nominal hip width
        right_y_offset = (x.xpos[self.right_foot_body][1] + 0.08) ** 2
        foot_placement_cost = 0.5 * (left_y_offset + right_y_offset)

        # penalize hip yaw — keeps legs pointing forward
        left_hip_yaw = x.qpos[8]  # leg_left_hip_yaw_joint
        right_hip_yaw = x.qpos[14]  # leg_right_hip_yaw_joint
        hip_yaw_cost = 0.5 * (left_hip_yaw ** 2 + right_hip_yaw ** 2)

        #reward stance extension — encourages the stance leg to be straightened during support phase
        left_knee = x.qpos[10]
        right_knee = x.qpos[16]
        reward_stance_extension = (
                left_contact * (-(left_knee - 0.15) ** 2) +
                right_contact * (-(right_knee - 0.15) ** 2)
        )

        #reward hip extension
        left_hip_pitch = x.qpos[9]
        right_hip_pitch = x.qpos[15]

        reward_stance_hip = (
                left_contact * (-(left_hip_pitch + 0.05) ** 2) +
                right_contact * (-(right_hip_pitch + 0.05) ** 2)
        )

        total = (healthy + height_penalty + forward
                 #+ balance_reward #it wants to stay balanced too much and ends up squatting instead of moving
                 + 2.0 * alternation
                 + 1.0 * swing_forward
                 + 3.0 * step_reward
                 + 2.0 * reward_stance_extension
                 + 2.0 * reward_stance_hip
                 - ctrl_cost
                 #- lateral_cost
                 #- lateral_swing_cost
                 #- foot_placement_cost
                 - hip_yaw_cost
                 )
        return -total

    def terminal_cost(self, x: mjx.Data) -> float:
        is_healthy = x.qpos[2] >= self.healthy_z_min
        return jnp.where(is_healthy, 0.0, 5.0)


##########OLD VERSION#########

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.task_base import Task


class HumanoidLocomotionTask(Task):
    '''
    Humanoid locomotion task with simple, effective reward shaping for forward walking.

    Key design:
    - Strong forward progress reward to drive movement
    - Foot lift reward to encourage active stepping
    - Posture penalties to maintain upright stance
    - Forward lean encouragement for natural gait
    '''

    def __init__(
            self,
            mj_model: mujoco.MjModel,
            rest_height: float = 0.85,
            target_speed: float = 0.2,
            forward_reward_weight: float = 10.0,
            step_reward_weight: float = 20.0,
            posture_cost_weight: float = 0.001,
            ctrl_cost_weight: float = 1e-3,
            healthy_reward: float = 1.0,
            healthy_z_min: float = 0.65,
            foot_contact_height: float = 0.04,
            step_swing_weight: float = 20.0,
            contact_placement_weight: float = 15.0,
            hip_swing_weight: float = 10.0,
    ):
        super().__init__(mj_model)

        # Core parameters
        self.rest_height = rest_height
        self.target_speed = target_speed
        self.forward_reward_weight = forward_reward_weight
        self.step_reward_weight = step_reward_weight
        self.posture_cost_weight = posture_cost_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_z_min = healthy_z_min
        self.foot_contact_height = foot_contact_height
        self.step_swing_weight = step_swing_weight
        # contact placement reward parameters and state
        self.contact_placement_weight = contact_placement_weight
        self.hip_swing_weight = hip_swing_weight
        # weight shift reward strength
        self.weight_shift_weight = 12.0
        # initialize previous contact flags and last contact x positions
        # start as True to avoid rewarding initial touchdown
        self._left_in_contact_prev = True
        self._right_in_contact_prev = True
        self._last_left_contact_x = None
        self._last_right_contact_x = None

        # Get foot body IDs
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

        # Nominal standing configuration (for posture cost)
        self.qstand = jnp.array([
            # Left leg
            0.0,    # hip roll
            0.0,    # hip yaw
            -0.20,  # hip pitch
            0.40,   # knee
            -0.20,  # ankle pitch
            0.0,    # ankle roll
            # Right leg
            0.0,    # hip roll
            0.0,    # hip yaw
            -0.20,  # hip pitch
            0.40,   # knee
            -0.20,  # ankle pitch
            0.0,    # ankle roll
        ], dtype=jnp.float32)

    def _is_healthy(self, x: mjx.Data) -> jax.Array:
        '''Check if robot is still upright.'''
        height = x.qpos[2]
        return height >= self.healthy_z_min

    def _get_pitch_roll(self, x: mjx.Data):
        '''Extract pitch and roll from quaternion.'''
        qw, qx, qy, qz = x.qpos[3], x.qpos[4], x.qpos[5], x.qpos[6]
        pitch = jnp.arcsin(2.0 * (qw * qy - qz * qx))
        roll = jnp.arctan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx**2 + qy**2),
        )
        return pitch, roll

    def forward_progress_reward(self, x: mjx.Data) -> jax.Array:
        '''
        Reward for moving forward.

        Directly incentivizes forward velocity to drive locomotion.
        '''
        forward_vel = x.qvel[0]

        # Gaussian reward peaked at target_speed
        reward = jnp.exp(
            -((forward_vel - self.target_speed) ** 2) / (2 * 0.25)
        )

        return self.forward_reward_weight * reward

    # The following legacy per-foot shaping terms are intentionally disabled now.
    # They were useful during early debugging, but the gait-phase reward below
    # already captures the stepping cycle more cleanly:
    # - step_reward
    # - swing_reward
    # - swing_forward_pos_reward
    # - knee_extension_reward

    def gait_phase_reward(self, x: mjx.Data) -> jax.Array: #gait =pattern of the legs when walking
        '''
        Reward the main walking phases explicitly:
        - one foot in stance while the other is in swing
        - swing foot moving forward
        - swing knee extending

        This turns the phase structure of a step into a dense reward signal.
        '''
        left_pos = x.xpos[self.left_foot_body]
        right_pos = x.xpos[self.right_foot_body]
        left_vel = x.cvel[self.left_foot_body][3:6]
        right_vel = x.cvel[self.right_foot_body][3:6]

        qpos_joints = x.qpos[7:7 + self.qstand.shape[0]] #first seven coordinates are x,y,z and quaternion (=orientation qw, qx, qy, qz)
        left_knee = qpos_joints[3]
        right_knee = qpos_joints[9]

        # smooth contact indicators in [0, 1]
        left_contact = jnp.clip((self.foot_contact_height - left_pos[2]) / self.foot_contact_height, 0.0, 1.0)
        right_contact = jnp.clip((self.foot_contact_height - right_pos[2]) / self.foot_contact_height, 0.0, 1.0)
        left_air = 1.0 - left_contact
        right_air = 1.0 - right_contact

        # Stance leg pushes the body forward
        left_push = left_contact * jnp.maximum(left_vel[0], 0.0)
        right_push = right_contact * jnp.maximum(right_vel[0], 0.0)

        stance_push_reward = left_push + right_push

        # phase-specific signals
        left_swing_phase = left_air * right_contact #ensure that when one is in the air the other is on the ground
        right_swing_phase = right_air * left_contact

        # Shift pelvis over the still foot before swing
        pelvis_y = x.qpos[1]

        left_support = jnp.exp(
            -20.0 * (pelvis_y - left_pos[1]) ** 2
        )

        right_support = jnp.exp(
            -20.0 * (pelvis_y - right_pos[1]) ** 2
        )

        weight_shift_reward = (
                left_swing_phase * right_support
                + right_swing_phase * left_support
        )

        base_x = x.qpos[0]
        left_forward = jnp.maximum(left_pos[0] - base_x, 0.0) #make sure the feet are moving forward
        right_forward = jnp.maximum(right_pos[0] - base_x, 0.0)

        left_knee_extension = jnp.maximum(self.qstand[3] - left_knee, 0.0) #ensure the knee extends during swing phase, but not penalize if it bends more than standing
        right_knee_extension = jnp.maximum(self.qstand[9] - right_knee, 0.0)

        left_forward_velocity = jnp.maximum(left_vel[0], 0.0)
        right_forward_velocity = jnp.maximum(right_vel[0], 0.0)

        # reward one-legged swing phases, not double support standing
        phase_balance = left_swing_phase + right_swing_phase
        double_support_penalty = left_contact * right_contact

        left_reward = left_swing_phase * (
            2.0 * left_forward
            + 1.5 * left_forward_velocity
            + 2.0 * left_knee_extension
        )
        right_reward = right_swing_phase * (
            2.0 * right_forward
            + 1.5 * right_forward_velocity
            + 2.0 * right_knee_extension
        )

        return 10.0 * (phase_balance + left_reward
                       #+ 2.0 * weight_shift_reward
                       + 1.5 * stance_push_reward + right_reward - 0.25 * double_support_penalty) #double support means if both feet stay on the ground penalty

    def posture_cost(self, x: mjx.Data) -> jax.Array:
        '''
        Penalize deviation from nominal standing posture.

        Encourages knees slightly bent (natural standing), discourages extreme poses.
        '''
        # Extract joint angles
        qpos_joints = x.qpos[7:7 + self.qstand.shape[0]]

        # Penalize squared deviation from standing posture
        cost = jnp.sum((qpos_joints - self.qstand) ** 2)

        return self.posture_cost_weight * cost

    def upright_reward(self, x: mjx.Data) -> jax.Array:
        '''
        Reward for maintaining upright posture (near-vertical orientation).

        Penalizes tilting and falling.
        '''
        height = x.qpos[2]
        pitch, roll = self._get_pitch_roll(x)

        # Height penalty: penalize dropping below rest height
        height_penalty = -2.0 * jnp.maximum(self.rest_height - height, 0.0) ** 2

        # Pitch penalty: penalize forward/backward tilting, but allow a stronger forward lean
        pitch_penalty = -0.25 * pitch ** 2

        # Roll penalty: penalize side tilting (more critical for balance)
        roll_penalty = -1.0 * roll ** 2

        # Forward lean encouragement: make the torso a bit more forward for walking
        desired_pitch = 0.18  # slightly more forward lean (~10 degrees)
        forward_lean = 2.5 * jnp.exp(-10.0 * (pitch - desired_pitch) ** 2)

        return height_penalty + pitch_penalty + roll_penalty + forward_lean

    def control_cost(self, u: jax.Array) -> jax.Array:
        '''Penalize large control inputs (encourage efficiency).'''
        return self.ctrl_cost_weight * jnp.sum(jnp.square(u))

    def healthy_penalty(self, x: mjx.Data) -> jax.Array:
        ''''''Penalize falling or unhealthy states.''''''
        is_healthy = self._is_healthy(x)

        penalty = jnp.where(
            is_healthy,
            self.healthy_reward,  # Positive reward for staying healthy
            -5.0,  # Large penalty for falling
        )

        return penalty

    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:
        '''
        Compute the running cost for one timestep.

        The optimizer minimizes this, so:
        - Positive values = rewards (costs we want to minimize)
        - Negative values = penalties (costs we want to maximize)
        '''

        # Reward forward progress
        forward = 2*self.forward_progress_reward(x)

        # Reward the actual gait phase structure directly
        gait_phase = 0.5*self.gait_phase_reward(x)

        # Maintain upright posture
        upright = 1.5*self.upright_reward(x)

        # Penalize control effort
        control = self.control_cost(u)

        # Penalize deviation from standing posture
        #posture = self.posture_cost(x)

        # Healthy/falling penalty
        healthy = 2*self.healthy_penalty(x)

        # Penalize lateral motion (walk straight)
        lateral_cost = 0.5 * x.qvel[1] ** 2

        # Penalize yaw rotation (don't spin)
        yaw_cost = 0.1 * x.qvel[5] ** 2

        # Total reward (negative sign because DIAL minimizes)
        total_reward = (
            forward
            + gait_phase
            + upright
            + healthy
            - control
            #- posture
            - lateral_cost
            - yaw_cost
        )

        # DIAL framework: minimize cost (so negate reward)
        return -total_reward

    def terminal_cost(self, x: mjx.Data) -> float:
                '''Cost when the episode terminates (robot falls).'''
        is_healthy = self._is_healthy(x)

        cost = jnp.where(
            is_healthy,
            0.0,  # No additional cost if healthy
            -self.healthy_reward,  # Penalty if fell
        )

        return cost

    def state_metrics(self, x: mjx.Data, u: jax.Array):
        '''
        Diagnostic metrics for debugging and analysis.
        '''
        left_pos = x.xpos[self.left_foot_body]
        right_pos = x.xpos[self.right_foot_body]
        left_lin_vel = x.cvel[self.left_foot_body][3:6]
        right_lin_vel = x.cvel[self.right_foot_body][3:6]

        pitch, roll = self._get_pitch_roll(x)

        metrics = {
            'height': x.qpos[2],
            'pitch': pitch,
            'roll': roll,
            'base_vx': x.qvel[0],
            'base_vy': x.qvel[1],
            'left_foot_z': left_pos[2],
            'right_foot_z': right_pos[2],
            'foot_height_diff': jnp.abs(left_pos[2] - right_pos[2]),
            'left_foot_vx': left_lin_vel[0],
            'right_foot_vx': right_lin_vel[0],
            'is_healthy': self._is_healthy(x),
        }

        return metrics

"""