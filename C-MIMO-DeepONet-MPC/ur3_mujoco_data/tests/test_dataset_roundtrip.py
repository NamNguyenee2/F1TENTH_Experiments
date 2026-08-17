"""Test: write a tiny dataset, reload from HDF5, check values match."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import h5py
import tempfile
from pathlib import Path
from ur3_mujoco.sim.mujoco_env import MujocoEnv
from ur3_mujoco.robot.ur3_robot import UR3Robot
from ur3_mujoco.dataset.kinematics_sampler import sample_one
from ur3_mujoco.dataset.kinematics_writer import write_dataset

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "scene.xml")
DUMMY_CONFIG = {
    "robot": {
        "model_path": "models/scene.xml",
        "joint_names": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        "joint_limits": [[-6.28, 6.28]] * 6,
        "q_velocity_limits": [[-3.14, 3.14]] * 6,
        "ee_site_name": "ee_site",
    },
    "dataset": {"stage": "test"},
}


def make_small_splits(robot, n=20):
    rng = np.random.default_rng(99)
    rows = [sample_one(robot, rng.uniform(-1.0, 1.0, 6), rng.uniform(-0.5, 0.5, 6))
            for _ in range(n)]
    data = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    return {"train": data, "val": data}


def test_roundtrip():
    env = MujocoEnv(MODEL_PATH)
    robot = UR3Robot(env)
    splits = make_small_splits(robot, n=30)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        path = tmp.name

    try:
        write_dataset(path, splits, {}, DUMMY_CONFIG, MODEL_PATH)
        with h5py.File(path, 'r') as f:
            assert "train" in f, "train split missing"
            assert "val" in f,   "val split missing"
            q_loaded = f["train"]["q"][:]
            assert q_loaded.shape == (30, 6), f"q shape wrong: {q_loaded.shape}"
            assert not np.any(np.isnan(q_loaded)), "NaN in reloaded q"
            y_loaded = f["train"]["y"][:]
            assert y_loaded.shape == (30, 15), f"y shape wrong: {y_loaded.shape}"
            # Values should match original
            assert np.allclose(q_loaded, splits["train"]["q"], atol=1e-10), "q mismatch"
        print("PASS test_roundtrip")
    finally:
        os.unlink(path)


def test_metadata_written():
    env = MujocoEnv(MODEL_PATH)
    robot = UR3Robot(env)
    splits = make_small_splits(robot, n=10)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        path = tmp.name
    try:
        write_dataset(path, splits, {}, DUMMY_CONFIG, MODEL_PATH)
        with h5py.File(path, 'r') as f:
            assert "metadata" in f
            assert "robot_name" in f["metadata"].attrs
            assert "joint_names" in f["metadata"].attrs
        print("PASS test_metadata_written")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_roundtrip()
    test_metadata_written()
    print("\nAll dataset roundtrip tests PASS")
