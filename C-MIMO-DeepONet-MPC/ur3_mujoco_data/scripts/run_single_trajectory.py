#!/usr/bin/env python3
"""Run a single trajectory and save to HDF5."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from pathlib import Path
from ur3_mujoco import UR3Model, PDController, TrajectoryGenerator, DataLogger
import mujoco

MODEL_PATH = Path(__file__).parent.parent / "models" / "scene.xml"
DATA_DIR = Path(__file__).parent.parent / "data"

def main():
    print("Loading UR3 model...")
    ur3 = UR3Model(str(MODEL_PATH))
    controller = PDController(ur3)
    traj_gen = TrajectoryGenerator()
    logger = DataLogger(ur3)

    print("Generating trajectory (joint 0 sine sweep)...")
    dt = ur3.model.opt.timestep
    traj = traj_gen.single_joint_sine(joint_idx=0, amplitude=0.5, frequency=0.2, duration=5.0, dt=dt)

    print(f"Trajectory: {len(traj['t'])} steps, duration={traj['t'][-1]:.2f}s")

    ur3.reset(traj['q'][0])
    logger.start_episode()

    for i in range(len(traj['t'])):
        q_ref = traj['q'][i]
        qdot_ref = traj['qdot'][i]
        qdotdot_ref = traj['qdotdot'][i]

        ctrl = controller.compute(ur3.get_q(), ur3.get_qdot(), q_ref, qdot_ref, qdotdot_ref)
        ur3.set_ctrl(ctrl)
        ur3.step()

        logger.log_step(ur3.data, ctrl, q_ref, qdot_ref, qdotdot_ref)

    ep_data = logger.get_episode_data()

    for k, v in ep_data.items():
        if np.any(np.isnan(v)):
            print(f"WARNING: NaN in {k}!")

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "single_trajectory.h5"
    metadata = {"model": "ur3", "traj_type": "single_joint_sine", "joint_idx": 0}
    logger.save_to_hdf5(str(out_path), "trajectory_000", metadata)
    print(f"Saved to {out_path}")

    reloaded = DataLogger.load_from_hdf5(str(out_path), "trajectory_000")
    print("Reload check:")
    for k, v in reloaded.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    print("Gate 5 PASS: Trajectory saved and reloaded successfully.")

if __name__ == "__main__":
    main()
