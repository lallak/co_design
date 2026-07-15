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
        is_healthy = (height >= self.healthy_z_min) & (jnp.abs(pitch) <= 0.6) & (jnp.abs(roll) <= 0.4)
        healthy = jnp.where(is_healthy, 1.0, -10.0)
        height_penalty = -3.0 * jnp.maximum(self.rest_height - height, 0.0) ** 2

        # 2. GO FORWARD — linear reward on forward velocity
        forward = 2.0 * jnp.maximum(x.qvel[0], 0.0)

        # 3. ALTERNATE FEET — reward one foot up while the other is down
        left_z  = x.xpos[self.left_foot_body][2]
        right_z = x.xpos[self.right_foot_body][2]
        foot_contact_height = 0.04
        left_contact  = jnp.clip((foot_contact_height - left_z)  / foot_contact_height, 0.0, 1.0)
        right_contact = jnp.clip((foot_contact_height - right_z) / foot_contact_height, 0.0, 1.0)
        # reward swing foot being lifted, but only during single support
        single_support = left_contact * (1.0 - right_contact) + right_contact * (1.0 - left_contact)
        swing_height = left_contact * right_z + right_contact * left_z  # height of the NON-contact foot
        alternation = single_support * jnp.clip(swing_height / 0.1, 0.0, 1.0)

        # reward swing foot moving forward while in the air
        left_air = 1.0 - left_contact
        right_air = 1.0 - right_contact
        swing_forward = (left_air * jnp.maximum(x.cvel[self.left_foot_body][3], 0.0) +
                         right_air * jnp.maximum(x.cvel[self.right_foot_body][3], 0.0))

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
        foot_placement_cost = 3.0 * (left_y_offset + right_y_offset)

        # penalize hip yaw — keeps legs pointing forward
        left_hip_yaw = x.qpos[8]  # leg_left_hip_yaw_joint
        right_hip_yaw = x.qpos[14]  # leg_right_hip_yaw_joint
        hip_yaw_cost = 5.0 * (left_hip_yaw ** 2 + right_hip_yaw ** 2)

        total = (healthy + height_penalty + forward
                 + 2.0 * alternation + 1.5 * swing_forward
                 - ctrl_cost - lateral_cost
                 - lateral_swing_cost - foot_placement_cost
                 - hip_yaw_cost)
        return -total

    def terminal_cost(self, x: mjx.Data) -> float:
        is_healthy = x.qpos[2] >= self.healthy_z_min
        return jnp.where(is_healthy, 0.0, -5.0)


##########OLD VERSION#########

"""import jax
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