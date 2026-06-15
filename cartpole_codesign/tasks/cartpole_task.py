import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from hydrax.task_base import Task


class CartpoleSwingupTask(Task):
    """
    CartPole swingup task for co-design.

    The pole starts hanging down. The goal is to swing it up
    and balance it at the inverted position.
    """

    def __init__(self, mj_model: mujoco.MjModel,
                 plan_horizon: float = 1.0,
                 num_knots: int = 10,
                 dt: float = 0.01):
        """
        Args:
            mj_model:      MuJoCo model (built from design parameters)
            plan_horizon:  How far ahead to plan, in seconds
            num_knots:     Number of spline control points
            dt:            Simulation timestep (should match XML)
        """
        super().__init__(mj_model)

        self.plan_horizon = plan_horizon
        self.num_knots = num_knots
        self.dt = dt

    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:
        """
        Cost at each timestep. Lower is better.

        State layout for CartPole in MuJoCo:
            x.qpos[0] = cart position
            x.qpos[1] = pole angle (0 = hanging down, π = upright)
            x.qvel[0] = cart velocity
            x.qvel[1] = pole angular velocity
        """
        cart_pos = x.qpos[0]
        pole_angle = x.qpos[1]

        # (1 - cos) is 0 when upright (angle=π), 2 when hanging (angle=0)
        # We use pole_angle directly — check your model's angle convention
        upright_cost = 1.0 - jnp.cos(pole_angle)

        # Keep cart near center
        cart_cost = 0.1 * jnp.square(cart_pos)

        # Penalize large control inputs
        control_cost = 0.01 * jnp.sum(jnp.square(u))

        return upright_cost + cart_cost + control_cost

    def terminal_cost(self, x: mjx.Data) -> float:
        """Cost at the final state (end of planning horizon)."""
        cart_pos = x.qpos[0]
        pole_angle = x.qpos[1]

        return (1.0 - jnp.cos(pole_angle)) + 0.1 * jnp.square(cart_pos)