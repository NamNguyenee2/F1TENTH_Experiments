"""Test: kinematics sample shapes, NaN, and y target composition."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pathlib import Path
from ur3_mujoco.sim.mujoco_env import MujocoEnv
from ur3_mujoco.robot.ur3_robot import UR3Robot
from ur3_mujoco.dataset.kinematics_sampler import sample_one

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "scene.xml")
EXPECTED_SHAPES = {
    "q":                (6,),
    "q_dot":            (6,),
    "sin_q":            (6,),
    "cos_q":            (6,),
    "x":                (18,),
    "ee_pos_world":     (3,),
    "ee_quat_world":    (4,),
    "ee_rotmat_world":  (9,),
    "ee_rot6d_world":   (6,),
    "ee_lin_vel_world": (3,),
    "ee_ang_vel_world": (3,),
    "J_pos_world":      (3, 6),
    "J_rot_world":      (3, 6),
    "J_world":          (6, 6),
    "y":                (15,),
}


def get_robot():
    env = MujocoEnv(MODEL_PATH)
    return UR3Robot(env)


def test_sample_shapes():
    robot = get_robot()
    q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    qdot = np.zeros(6)
    sample = sample_one(robot, q, qdot)
    for field, expected in EXPECTED_SHAPES.items():
        assert field in sample, f"Missing field: {field}"
        assert sample[field].shape == expected, \
            f"{field}: got {sample[field].shape}, expected {expected}"
    print("PASS test_sample_shapes")


def test_no_nan():
    robot = get_robot()
    rng = np.random.default_rng(42)
    for _ in range(10):
        q = rng.uniform(-1.0, 1.0, 6)
        qdot = rng.uniform(-1.0, 1.0, 6)
        sample = sample_one(robot, q, qdot)
        for field, v in sample.items():
            assert not np.any(np.isnan(v)), f"NaN in {field}"
            assert not np.any(np.isinf(v)), f"Inf in {field}"
    print("PASS test_no_nan")


def test_y_composition():
    robot = get_robot()
    q = np.array([0.1, -1.2, 1.3, -1.4, -1.5, 0.2])
    qdot = np.array([0.05, -0.1, 0.2, -0.05, 0.1, -0.2])
    sample = sample_one(robot, q, qdot)
    y = sample["y"]
    assert np.allclose(y[:3],  sample["ee_pos_world"],     atol=1e-10), "y[:3] != ee_pos"
    assert np.allclose(y[3:9], sample["ee_rot6d_world"],   atol=1e-10), "y[3:9] != ee_rot6d"
    assert np.allclose(y[9:12], sample["ee_lin_vel_world"], atol=1e-10), "y[9:12] != lin_vel"
    assert np.allclose(y[12:], sample["ee_ang_vel_world"], atol=1e-10), "y[12:] != ang_vel"
    print("PASS test_y_composition")


def test_x_composition():
    robot = get_robot()
    q = np.array([0.3, -0.5, 0.7, -0.9, 1.1, -1.3])
    qdot = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    sample = sample_one(robot, q, qdot)
    x = sample["x"]
    assert np.allclose(x[:6],   np.sin(q),  atol=1e-10), "x[:6] != sin(q)"
    assert np.allclose(x[6:12], np.cos(q),  atol=1e-10), "x[6:12] != cos(q)"
    assert np.allclose(x[12:],  qdot,       atol=1e-10), "x[12:] != qdot"
    print("PASS test_x_composition")


if __name__ == "__main__":
    test_sample_shapes()
    test_no_nan()
    test_y_composition()
    test_x_composition()
    print("\nAll kinematics shape tests PASS")
