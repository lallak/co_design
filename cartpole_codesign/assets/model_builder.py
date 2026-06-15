import mujoco
import numpy as np

TEMPLATE_PATH = "cartpole_codesign/assets/cart_pole_scene.xml"


def build_cartpole_model(theta: np.ndarray) -> mujoco.MjModel:
    """
    Build a MuJoCo model from design parameters.

    Args:
        theta: [pole_length, pole_mass, cart_mass]

    Returns:
        mj_model: MuJoCo model with the specified design
    """
    pole_length, pole_mass, cart_mass = theta

    with open(TEMPLATE_PATH, 'r') as f:
        template = f.read()

    xml_string = template.format(
        pole_length=pole_length,
        pole_mass=pole_mass,
        cart_mass=cart_mass
    )

    return mujoco.MjModel.from_xml_string(xml_string)