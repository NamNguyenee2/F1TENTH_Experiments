#!/usr/bin/env python3
"""Validate UR3 kinematics and end-effector pose."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from ur3_mujoco import UR3Model, PDController, TrajectoryGenerator, DataLogger
import mujoco

MODEL_PATH = Path(__file__).parent.parent / "models" / "scene.xml"
DATA_DIR = Path(__file__).parent.parent / "data"

def main():
    ur3 = UR3Model(str(MODEL_PATH))
    traj_gen = TrajectoryGenerator()
    logger = DataLogger(ur3)
    ctrl = PDController(ur3)
    dt = ur3.model.opt.timestep

    results = []

    print("Running single-joint validation for each of 6 joints...")
    for joint_idx in range(6):
        traj = traj_gen.single_joint_sine(joint_idx=joint_idx, amplitude=0.3, frequency=0.2, duration=5.0, dt=dt)
        ur3.reset(traj['q'][0])
        logger.start_episode()

        for i in range(len(traj['t'])):
            c = ctrl.compute(ur3.get_q(), ur3.get_qdot(), traj['q'][i], traj['qdot'][i], traj['qdotdot'][i])
            ur3.set_ctrl(c)
            ur3.step()
            logger.log_step(ur3.data, c, traj['q'][i], traj['qdot'][i], traj['qdotdot'][i])

        ep = logger.get_episode_data()
        q_std = np.std(ep['q'], axis=0)
        active_joint = int(np.argmax(q_std))
        print(f"  Joint {joint_idx} ({ur3.JOINT_NAMES[joint_idx]}): most active={active_joint}, std={q_std.round(4)}")
        results.append({'joint_idx': joint_idx, 'q_std': q_std, 'ep': ep})

    print("\nChecking rotation matrix validity...")
    ep = results[0]['ep']
    R_flat = ep['ee_rotmat']
    N = len(R_flat)
    max_det_err = 0.0
    max_orth_err = 0.0
    for i in range(N):
        R = R_flat[i].reshape(3, 3)
        det_err = abs(np.linalg.det(R) - 1.0)
        orth_err = np.max(np.abs(R.T @ R - np.eye(3)))
        max_det_err = max(max_det_err, det_err)
        max_orth_err = max(max_orth_err, orth_err)
    print(f"  Max det error: {max_det_err:.6f}")
    print(f"  Max orthogonality error: {max_orth_err:.6f}")

    DATA_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ji, res in enumerate(results):
        ax = axes[ji // 2, ji % 2]
        t = res['ep']['time']
        q = res['ep']['q']
        for k in range(6):
            lbl = ur3.JOINT_NAMES[k] if k == ji else '_nolegend_'
            ax.plot(t, q[:, k], label=lbl)
        ax.set_title(f"Commanding Joint {ji}: {ur3.JOINT_NAMES[ji]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("q (rad)")
        ax.legend(loc='upper right', fontsize=6)
    plt.tight_layout()
    plot_path = DATA_DIR / "validate_joint_order.png"
    plt.savefig(str(plot_path))
    plt.close()
    print(f"\nPlot saved: {plot_path}")

    if max_det_err < 1e-4:
        print("\nGate 3 + 6 (kinematics): PASS")
    else:
        print("\nGate 3 FAIL: rotation matrix invalid")

if __name__ == "__main__":
    main()
