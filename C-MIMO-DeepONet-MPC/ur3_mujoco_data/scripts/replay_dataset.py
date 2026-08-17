#!/usr/bin/env python3
"""Replay a saved trajectory in MuJoCo viewer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import argparse
import h5py
import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path
from ur3_mujoco import UR3Model

MODEL_PATH = Path(__file__).parent.parent / "models" / "scene.xml"
DATA_DIR = Path(__file__).parent.parent / "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--trajectory', type=str, default='trajectory_000')
    args = parser.parse_args()

    h5_path = args.file or str(DATA_DIR / "single_trajectory.h5")

    print(f"Loading {args.trajectory} from {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        traj = {k: f[args.trajectory][k][:] for k in f[args.trajectory].keys()}

    print(f"Trajectory: {len(traj['time'])} steps, duration={traj['time'][-1]:.2f}s")

    ur3 = UR3Model(str(MODEL_PATH))
    q_traj = traj['q']
    ur3.reset(q_traj[0])

    print("Launching viewer. Replaying trajectory...")
    with mujoco.viewer.launch_passive(ur3.model, ur3.data) as viewer:
        for i in range(len(q_traj)):
            for ji, jname in enumerate(ur3.JOINT_NAMES):
                jid = mujoco.mj_name2id(ur3.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                addr = ur3.model.jnt_qposadr[jid]
                ur3.data.qpos[addr] = q_traj[i, ji]
            mujoco.mj_forward(ur3.model, ur3.data)
            viewer.sync()
            time.sleep(ur3.model.opt.timestep)

    print("Replay complete.")

if __name__ == "__main__":
    main()
