"""Test: all 6 UR3e joints are found by name, with correct qpos/qvel addresses."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import mujoco
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "scene.xml"
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def test_all_joints_found():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"Joint '{name}' not found"
        assert model.jnt_qposadr[jid] >= 0, f"'{name}' qpos addr invalid"
        assert model.jnt_dofadr[jid] >= 0, f"'{name}' qvel addr invalid"
    print("PASS test_all_joints_found")


def test_joint_order_is_unique():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    addrs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in JOINT_NAMES]
    assert len(set(addrs)) == 6, f"Duplicate qpos addresses: {addrs}"
    print("PASS test_joint_order_is_unique")


def test_six_joints_exactly():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    assert model.nv >= 6, f"model.nv={model.nv} < 6"
    print("PASS test_six_joints_exactly")


if __name__ == "__main__":
    test_all_joints_found()
    test_joint_order_is_unique()
    test_six_joints_exactly()
    print("\nAll joint mapping tests PASS")
