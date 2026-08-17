"""Test: ee_lin_vel = J_pos @ q_dot and ee_ang_vel = J_rot @ q_dot."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pathlib import Path
from ur3_mujoco.sim.mujoco_env import MujocoEnv
from ur3_mujoco.robot.ur3_robot import UR3Robot
from ur3_mujoco.dataset.kinematics_sampler import sample_one

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "scene.xml")


def test_velocity_jacobian_consistency():
    """ee_lin_vel and ee_ang_vel must exactly equal J @ q_dot (they are derived that way)."""
    env = MujocoEnv(MODEL_PATH)
    robot = UR3Robot(env)
    rng = np.random.default_rng(7)

    max_lin_err = 0.0
    max_ang_err = 0.0
    for _ in range(50):
        q = rng.uniform(-1.5, 1.5, 6)
        qdot = rng.uniform(-2.0, 2.0, 6)
        s = sample_one(robot, q, qdot)

        lin_expected = s["J_pos_world"] @ qdot
        ang_expected = s["J_rot_world"] @ qdot
        lin_err = float(np.max(np.abs(s["ee_lin_vel_world"] - lin_expected)))
        ang_err = float(np.max(np.abs(s["ee_ang_vel_world"] - ang_expected)))
        max_lin_err = max(max_lin_err, lin_err)
        max_ang_err = max(max_ang_err, ang_err)

    assert max_lin_err < 1e-10, f"lin_vel != J_pos@qdot, max_err={max_lin_err:.2e}"
    assert max_ang_err < 1e-10, f"ang_vel != J_rot@qdot, max_err={max_ang_err:.2e}"
    print(f"PASS test_velocity_jacobian_consistency "
          f"(lin_err={max_lin_err:.2e}, ang_err={max_ang_err:.2e})")


def test_zero_velocity_at_home():
    """With q_dot=0, ee velocities must be zero."""
    env = MujocoEnv(MODEL_PATH)
    robot = UR3Robot(env)
    q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    qdot = np.zeros(6)
    s = sample_one(robot, q, qdot)
    assert np.allclose(s["ee_lin_vel_world"], 0.0, atol=1e-12), "Non-zero lin_vel at zero qdot"
    assert np.allclose(s["ee_ang_vel_world"], 0.0, atol=1e-12), "Non-zero ang_vel at zero qdot"
    print("PASS test_zero_velocity_at_home")


def test_jacobian_shape():
    env = MujocoEnv(MODEL_PATH)
    robot = UR3Robot(env)
    q = np.zeros(6)
    qdot = np.zeros(6)
    s = sample_one(robot, q, qdot)
    assert s["J_pos_world"].shape == (3, 6)
    assert s["J_rot_world"].shape == (3, 6)
    assert s["J_world"].shape == (6, 6)
    print("PASS test_jacobian_shape")


if __name__ == "__main__":
    test_velocity_jacobian_consistency()
    test_zero_velocity_at_home()
    test_jacobian_shape()
    print("\nAll velocity consistency tests PASS")
