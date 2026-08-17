#!/usr/bin/env python3
"""Replay a trajectory from the kinematics dataset live in the MuJoCo viewer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import time
import h5py
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "scene.xml")
DEFAULT_DATASET = str(Path(__file__).parent.parent / "data" / "debug" / "ur3_kinematics_debug.h5")

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def load_trajectory(dataset_path: str, group_path: str) -> dict:
    with h5py.File(dataset_path, 'r') as f:
        if group_path not in f:
            available = []
            def _collect(name, obj):
                if isinstance(obj, h5py.Group) and 'q' in obj:
                    available.append(name)
            f.visititems(_collect)
            print(f"Group '{group_path}' not found. Available trajectories:")
            for p in available:
                print(f"  {p}")
            sys.exit(1)
        grp = f[group_path]
        return {k: grp[k][:] for k in grp.keys()}


def main():
    parser = argparse.ArgumentParser(
        description="Replay a kinematics dataset trajectory in the MuJoCo viewer.")
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        help='Path to HDF5 dataset')
    parser.add_argument('--group', default='debug_trajectories/single_joint_sine/traj_000',
                        help='HDF5 group path of the trajectory to replay')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (e.g. 2.0 = 2x faster)')
    parser.add_argument('--loop', action='store_true',
                        help='Loop the trajectory continuously')
    args = parser.parse_args()

    print(f"Dataset : {args.dataset}")
    print(f"Group   : {args.group}")
    traj = load_trajectory(args.dataset, args.group)

    q_traj = traj['q']          # [N, 6]
    t_arr  = traj.get('time', np.arange(len(q_traj)) * 0.002)
    N = len(q_traj)
    dt = float(np.mean(np.diff(t_arr))) if N > 1 else 0.002

    print(f"Steps   : {N}  |  Duration: {t_arr[-1]:.2f}s  |  dt: {dt*1000:.1f}ms")
    print(f"Speed   : {args.speed}x  |  Loop: {args.loop}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    # Build name→qpos_addr map
    qpos_addrs = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_addrs.append(int(model.jnt_qposadr[jid]))

    def set_q(i):
        for addr, val in zip(qpos_addrs, q_traj[i]):
            data.qpos[addr] = val
        mujoco.mj_forward(model, data)

    # Set initial pose before opening viewer
    set_q(0)

    print("\nOpening viewer... (close the window to stop)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.4
        viewer.cam.azimuth  = 135
        viewer.cam.elevation = -20

        play_dt = dt / args.speed
        run = True
        while run and viewer.is_running():
            for i in range(N):
                if not viewer.is_running():
                    run = False
                    break
                set_q(i)
                viewer.sync()
                time.sleep(play_dt)

            if not args.loop:
                break

            # Brief pause before looping
            time.sleep(0.5)

    print("Replay complete.")


if __name__ == "__main__":
    main()
