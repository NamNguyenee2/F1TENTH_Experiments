#!/usr/bin/env python3
"""Quick visualization of tiny dataset."""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dataset_path = Path("data/tiny/ur3_kinematics_tiny.h5")

with h5py.File(dataset_path, 'r') as f:
    # Get train data
    q_train = f['train']['q'][:]
    ee_pos = f['train']['ee_pos_world'][:]
    ee_vel = f['train']['ee_lin_vel_world'][:]
    
    # Get debug trajectories
    traj_single = f['debug_trajectories/single_joint_sine/traj_000']
    q_traj = traj_single['q'][:]
    t_traj = traj_single['time'][:]
    ee_pos_traj = traj_single['ee_pos_world'][:]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Joint space samples (train)
    ax = axes[0, 0]
    for i in range(min(10, len(q_train))):
        ax.scatter(q_train[i, 0], q_train[i, 1], s=50, alpha=0.6)
    ax.set_xlabel("Joint 0 (rad)")
    ax.set_ylabel("Joint 1 (rad)")
    ax.set_title(f"Joint Space Samples (train, N={len(q_train)})")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: End-effector positions (train)
    ax = axes[0, 1]
    ax.scatter(ee_pos[:, 0], ee_pos[:, 1], s=50, c=ee_pos[:, 2], cmap='viridis', alpha=0.7)
    ax.set_xlabel("EE X (m)")
    ax.set_ylabel("EE Y (m)")
    ax.set_title(f"End-Effector Positions (train, N={len(ee_pos)})")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("EE Z (m)")
    
    # Plot 3: Single-joint sine trajectory
    ax = axes[1, 0]
    ax.plot(t_traj, q_traj[:, 0], label='J0', linewidth=2)
    ax.plot(t_traj, q_traj[:, 1], label='J1', linewidth=2)
    ax.plot(t_traj, q_traj[:, 2], label='J2', linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Position (rad)")
    ax.set_title("Single-Joint Sine Trajectory (first 3 joints)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: EE trajectory 3D projection
    ax = axes[1, 1]
    ax.plot(ee_pos_traj[:, 0], ee_pos_traj[:, 1], linewidth=2, alpha=0.7, color='blue')
    ax.scatter(ee_pos_traj[0, 0], ee_pos_traj[0, 1], s=200, marker='o', color='green', label='Start', zorder=5)
    ax.scatter(ee_pos_traj[-1, 0], ee_pos_traj[-1, 1], s=200, marker='s', color='red', label='End', zorder=5)
    ax.set_xlabel("EE X (m)")
    ax.set_ylabel("EE Y (m)")
    ax.set_title("End-Effector Trajectory (XY plane)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = Path("data/tiny/visualization.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    print(f"[OK] Saved: {out_path}")
    
    # Print stats
    print("\n" + "="*60)
    print("TINY DATASET STATS")
    print("="*60)
    print(f"\nTrain split:")
    print(f"  Samples: {len(q_train)}")
    print(f"  Joint pos range: [{q_train.min():.3f}, {q_train.max():.3f}] rad")
    print(f"  EE pos range X: [{ee_pos[:, 0].min():.3f}, {ee_pos[:, 0].max():.3f}] m")
    print(f"  EE pos range Y: [{ee_pos[:, 1].min():.3f}, {ee_pos[:, 1].max():.3f}] m")
    print(f"  EE pos range Z: [{ee_pos[:, 2].min():.3f}, {ee_pos[:, 2].max():.3f}] m")
    print(f"  EE vel range: [{ee_vel.min():.3f}, {ee_vel.max():.3f}] m/s")
    
    print(f"\nSingle-joint sine trajectory (traj_000):")
    print(f"  Duration: {t_traj[-1]:.2f} s")
    print(f"  Steps: {len(t_traj)}")
    print(f"  EE displacement: {np.linalg.norm(ee_pos_traj[-1] - ee_pos_traj[0]):.4f} m")

print("\n[OK] Visualization complete!")
