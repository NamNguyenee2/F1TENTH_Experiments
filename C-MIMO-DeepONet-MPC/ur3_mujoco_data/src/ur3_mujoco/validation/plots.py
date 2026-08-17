import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import h5py


def plot_debug_trajectories(dataset_path: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, 'r') as f:
        # ── Single-joint sine plots ─────────────────────────────────
        sjs_path = "debug_trajectories/single_joint_sine"
        if sjs_path in f:
            grp = f[sjs_path]
            keys = sorted(grp.keys())
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            for i, key in enumerate(keys[:6]):
                t = grp[key]["time"][:]
                q = grp[key]["q"][:]
                for j in range(6):
                    lw = 2.0 if j == i else 0.8
                    axes[i].plot(t, q[:, j], lw=lw, label=f"j{j}" if j == i else "_")
                axes[i].set_title(f"Commanding joint {i}")
                axes[i].set_xlabel("time (s)")
                axes[i].set_ylabel("q (rad)")
                axes[i].legend(fontsize=7)
            plt.suptitle("Single-Joint Sine Validation")
            plt.tight_layout()
            plt.savefig(str(out / "single_joint_sine.png"), dpi=100)
            plt.close()
            print(f"  Saved: single_joint_sine.png")

        # ── Multi-joint trajectory plots ────────────────────────────
        mjs_path = "debug_trajectories/multi_joint_sine"
        if mjs_path in f:
            grp = f[mjs_path]
            key = sorted(grp.keys())[0]
            t = grp[key]["time"][:]
            q = grp[key]["q"][:]
            ee_pos = grp[key]["ee_pos_world"][:]
            ee_vel_jac = grp[key]["ee_lin_vel_world"][:]
            ee_vel_fd = np.gradient(ee_pos, t, axis=0)

            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            labels = ['x', 'y', 'z']
            for j in range(6):
                axes[0, 0].plot(t, q[:, j], label=f"j{j}")
            axes[0, 0].set_title("q over time (multi-joint sine)")
            axes[0, 0].set_xlabel("time (s)"); axes[0, 0].set_ylabel("rad")
            axes[0, 0].legend(fontsize=7)

            for d in range(3):
                axes[0, 1].plot(t, ee_pos[:, d], label=labels[d])
            axes[0, 1].set_title("ee_pos_world over time")
            axes[0, 1].set_xlabel("time (s)"); axes[0, 1].set_ylabel("m")
            axes[0, 1].legend()

            for d in range(3):
                axes[1, 0].plot(t, ee_vel_jac[:, d], label=f"Jac {labels[d]}")
                axes[1, 0].plot(t, ee_vel_fd[:, d], '--', alpha=0.6, label=f"FD {labels[d]}")
            axes[1, 0].set_title("EE lin vel: Jacobian vs FD")
            axes[1, 0].legend(fontsize=6)

            err = np.abs(ee_vel_jac - ee_vel_fd)
            axes[1, 1].plot(t, err)
            axes[1, 1].set_title(f"Vel error |Jac-FD| (mean={err.mean():.4f})")
            axes[1, 1].set_xlabel("time (s)")

            plt.tight_layout()
            plt.savefig(str(out / "multi_joint_trajectory.png"), dpi=100)
            plt.close()
            print(f"  Saved: multi_joint_trajectory.png")

        # ── Coverage histograms ─────────────────────────────────────
        if "train" in f:
            q = f["train"]["q"][:]
            ee_pos = f["train"]["ee_pos_world"][:]
            qdot = f["train"]["q_dot"][:]

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for j in range(6):
                axes[j // 3, j % 3].hist(q[:, j], bins=50, edgecolor='k', alpha=0.7)
                axes[j // 3, j % 3].set_title(f"q[{j}] distribution")
                axes[j // 3, j % 3].set_xlabel("rad")
            plt.suptitle("Joint-space coverage (train)")
            plt.tight_layout()
            plt.savefig(str(out / "q_coverage.png"), dpi=100)
            plt.close()

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for d, l in enumerate(['x', 'y', 'z']):
                axes[d].hist(ee_pos[:, d], bins=50, edgecolor='k', alpha=0.7)
                axes[d].set_title(f"ee_pos {l}")
                axes[d].set_xlabel("m")
            plt.suptitle("EE position distribution (train)")
            plt.tight_layout()
            plt.savefig(str(out / "ee_pos_distribution.png"), dpi=100)
            plt.close()

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for j in range(6):
                axes[j // 3, j % 3].hist(qdot[:, j], bins=50, edgecolor='k', alpha=0.7)
                axes[j // 3, j % 3].set_title(f"q_dot[{j}] distribution")
                axes[j // 3, j % 3].set_xlabel("rad/s")
            plt.suptitle("q_dot coverage (train)")
            plt.tight_layout()
            plt.savefig(str(out / "qdot_coverage.png"), dpi=100)
            plt.close()

            print(f"  Saved: q_coverage.png, ee_pos_distribution.png, qdot_coverage.png")
