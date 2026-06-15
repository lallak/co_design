import mujoco
import numpy as np

TEMPLATE_PATH = "hopper_codesign/assets/hopper_scene.xml"


def build_hopper_model(theta: np.ndarray) -> mujoco.MjModel:
    """
    Build a MuJoCo hopper model from design parameters.

    Args:
        theta: [thigh_length, leg_length, rho]
            - rho: linear density (kg/m), mass = rho * length
    Returns:
        mj_model: MuJoCo model with the specified design
    """
    DAMPING     = 0.5   # fixed joint damping
    FOOT_LENGTH = 0.1   # fixed foot geometry

    thigh_length, leg_length, rho = theta

    # Enforce minimums
    thigh_length = max(thigh_length, 0.1)
    leg_length   = max(leg_length,   0.1)
    rho          = max(rho,          0.1)

    # Derive masses from geometry (uniform linear density)
    thigh_mass = max(rho * thigh_length, 0.1)
    leg_mass   = max(rho * leg_length,   0.1)
    foot_mass  = max(rho * FOOT_LENGTH,  0.05)

    with open(TEMPLATE_PATH, 'r') as f:
        template = f.read()

    xml_string = template.format(
        thigh_length=thigh_length,
        leg_length=leg_length,
        thigh_length_half=thigh_length / 2,
        leg_length_half=leg_length / 2,
        thigh_mass=thigh_mass,
        leg_mass=leg_mass,
        foot_mass=foot_mass,
        damping=DAMPING,
        torso_height=0.5 + thigh_length + leg_length,
    )

    return mujoco.MjModel.from_xml_string(xml_string)


def get_rest_height(theta: np.ndarray) -> float:
    """Hauteur réelle du torso au repos pour ce design."""
    thigh_length, leg_length = theta[0], theta[1]
    return 0.5 + thigh_length + leg_length