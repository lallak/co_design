import os
import tempfile
import mujoco
import numpy as np

#SCENE_PATH = "humanoid_codesign/assets/bhl_biped_scene.xml"
SCENE_PATH = os.path.join(os.path.dirname(__file__), "bhl_biped_scene.xml")

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

    # MuJoCo needs a real file path on disk to resolve <include> and
    # meshdir relative paths — from_xml_string has no directory context,
    # which is what was causing the mangled mesh paths.
    scene_dir = os.path.dirname(SCENE_PATH)
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", dir=scene_dir, delete=False
    ) as tmp:
        tmp.write(xml)
        tmp_path = tmp.name

    try:
        return mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.remove(tmp_path)

    return mujoco.MjModel.from_xml_string(xml)

def get_rest_height(theta: np.ndarray) -> float:
    thigh_length, shank_length, _ = theta
    return thigh_length + shank_length - 0.79