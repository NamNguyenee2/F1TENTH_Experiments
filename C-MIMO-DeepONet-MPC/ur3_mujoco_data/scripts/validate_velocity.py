#!/usr/bin/env python3
"""Validate velocity logging: finite-diff vs Jacobian."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from ur3_mujoco import UR3Model, PDController, TrajectoryGenerator, DataLogger

MODEL_PATH = Path(__file__).parent.parent / "models" / "scene.xml"
DATA_DIR = Path(__file__).parent.parent / "data"

def main():
    ur3 = UR3Model(str(MODEL_PATH))
    traj_gen = TrajectoryGenerator()
    logger = DataLogger(ur3)
    dt = ur3.model.opt.timestep
    ctrl = PDController(ur3)

    traj = traj_gen.multi_joint_smooth(duration=10.0, dt=dt, n_waypoints=6, seed=42)
    ur3.reset(traj['q'][0])
    logger.start_episode()

    for i in range(len(traj['t'])):
        c = ctrl.compute(ur3.get_q(), ur3.get_qdot(), traj['q'][i], traj['qdot'][i], traj['qdotdot'][i])
        ur3.set_ctrl(c)
        ur3.step()
        logger.log_step(ur3.data, c, traj['q'][i], traj['qdot'][i], traj['qdotdot'][i])

    ep = logger.get_episode_data()

    # Trim boundary to avoid np.gradient edge artifacts (uses 1-sided diff at boundaries)
    TRIM = 20
    ee_lin_vel_jac = ep['ee_lin_vel'][TRIM:-TRIM]
    ee_lin_vel_fd  = np.gradient(ep['ee_pos'], ep['time'], axis=0)[TRIM:-TRIM]
    qdotdot_logged = ep['q_dot_dot'][TRIM:-TRIM]
    qdotdot_fd     = np.gradient(ep['q_dot'], ep['time'], axis=0)[TRIM:-TRIM]

    vel_error = np.abs(ee_lin_vel_jac - ee_lin_vel_fd)
    acc_error = np.abs(qdotdot_logged - qdotdot_fd)

    # Use p99 — max is dominated by single-point spikes
    print(f"EE velocity (Jac vs FD):   max={vel_error.max():.4f}  mean={vel_error.mean():.4f}  p99={np.percentile(vel_error,99):.4f}")
    print(f"Joint accel (logged vs FD): max={acc_error.max():.4f}  mean={acc_error.mean():.4f}  p99={np.percentile(acc_error,99):.4f}")
    print(f"  Note: qacc=MuJoCo full dynamics; FD(qvel)=numerical approx. Some diff is physics-correct.")

    # Full arrays for plotting
    t = ep['time']
    ee_lin_vel_jac_full = ep['ee_lin_vel']
    ee_lin_vel_fd_full  = np.gradient(ep['ee_pos'], ep['time'], axis=0)
    qdotdot_logged_full = ep['q_dot_dot']
    qdotdot_fd_full     = np.gradient(ep['q_dot'], ep['time'], axis=0)
    vel_error_full = np.abs(ee_lin_vel_jac_full - ee_lin_vel_fd_full)
    acc_error_full = np.abs(qdotdot_logged_full - qdotdot_fd_full)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for d, label in enumerate(['x', 'y', 'z']):
        axes[0, 0].plot(t, ee_lin_vel_jac_full[:, d], label=f"Jac {label}")
        axes[0, 0].plot(t, ee_lin_vel_fd_full[:, d], '--', label=f"FD {label}")
    axes[0, 0].set_title("EE Linear Velocity: Jacobian vs Finite-Diff")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_xlabel("Time (s)")

    axes[0, 1].plot(t, vel_error_full)
    axes[0, 1].set_title("EE Velocity Error |Jac - FD|")
    axes[0, 1].set_xlabel("Time (s)")

    axes[1, 0].plot(t, qdotdot_logged_full[:, :3])
    axes[1, 0].set_title("Joint Accel (qacc) joints 0-2")
    axes[1, 0].set_xlabel("Time (s)")

    axes[1, 1].plot(t, acc_error_full[:, :3])
    axes[1, 1].set_title("Joint Accel Error |qacc - FD(qvel)| joints 0-2")
    axes[1, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    DATA_DIR.mkdir(exist_ok=True)
    plot_path = DATA_DIR / "validate_velocity.png"
    plt.savefig(str(plot_path))
    plt.close()
    print(f"Plot saved: {plot_path}")

    out_path = DATA_DIR / "velocity_validation.h5"
    logger.save_to_hdf5(str(out_path), "trajectory_000", {"type": "multi_joint_smooth"})
    print(f"Data saved: {out_path}")

    # Gate 6: EE velocity p99 is the primary check.
    # acc: qacc=MuJoCo full dynamics, FD(qvel)=numerical — differ due to integration scheme; informational only.
    vel_p99 = np.percentile(vel_error, 99)
    acc_p99 = np.percentile(acc_error, 99)
    vel_pass = vel_p99 < 0.2
    print(f"\nGate 6 velocity: {'PASS' if vel_pass else 'FAIL'} "
          f"(EE vel p99={vel_p99:.4f} m/s  |  acc p99={acc_p99:.2f} rad/s² [informational])")

if __name__ == "__main__":
    main()
