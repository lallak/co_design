import mujoco
import numpy as np

SCENE_PATH = "berkeley_humanoid_lite_biped_scene.xml"

HIP_HEIGHT  = 0.542631 #adapt to rory's robot
FOOT_LENGTH = 0.22


def build_humanoid_model(theta: np.ndarray) -> mujoco.MjModel:
    """
    Args:
        theta: [thigh_length, shank_length, rho]
    """
    thigh_length, shank_length, rho = theta

    thigh_length = max(thigh_length, 0.05)
    shank_length = max(shank_length, 0.05)
    rho          = max(rho,          0.1)

    thigh_mass = max(rho * thigh_length, 0.05)
    shank_mass = max(rho * shank_length, 0.05)
    foot_mass  = max(rho * FOOT_LENGTH,  0.05)

    with open(SCENE_PATH, "r") as f:
        xml = f.read()

    xml = xml.format(
        thigh_length=thigh_length,
        shank_length=shank_length,
        thigh_mass=thigh_mass,
        shank_mass=shank_mass,
        foot_mass=foot_mass,
    )

    return mujoco.MjModel.from_xml_string(xml)


def get_rest_height(theta: np.ndarray) -> float:
    thigh_length, shank_length, _ = theta
    return HIP_HEIGHT + max(thigh_length, 0.05) + max(shank_length, 0.05)