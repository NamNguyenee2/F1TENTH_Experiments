import h5py
import time
from pathlib import Path


def write_report(dataset_path: str, errors: list, warnings: list, output_path: str):
    passed = len(errors) == 0
    with h5py.File(dataset_path, 'r') as f:
        meta = dict(f["metadata"].attrs)
        split_sizes = {}
        for s in ["train", "val", "test_random", "test_trajectory", "test_boundary"]:
            if s in f and "q" in f[s]:
                split_sizes[s] = int(f[s]["q"].shape[0])
            else:
                split_sizes[s] = 0

    lines = [
        "# UR3e Kinematics Dataset — Validation Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Dataset:** `{dataset_path}`",
        f"**Result:** {'✅ PASS' if passed else '❌ FAIL'}",
        "",
        "## Dataset Info",
        f"- Robot: {meta.get('robot_name', '?')}",
        f"- MuJoCo version: {meta.get('mujoco_version', '?')}",
        f"- Model hash: {meta.get('model_hash', '?')}",
        f"- EE site: `{meta.get('ee_site_name', '?')}`",
        f"- EE site definition: {meta.get('ee_site_definition', '?')}",
        f"- Stage: `{meta.get('sampling_strategy', '?')}`",
        f"- Generated: {meta.get('date_generated', '?')}",
        "",
        "## Split Sizes",
    ]
    for s, n in split_sizes.items():
        lines.append(f"- `/{s}`: {n} samples")

    lines += ["", "## Validation Errors"]
    lines += [f"- ❌ {e}" for e in errors] if errors else ["None ✅"]
    lines += ["", "## Warnings"]
    lines += [f"- ⚠️ {w}" for w in warnings] if warnings else ["None ✅"]
    lines += [
        "",
        "## Data Conventions",
        "| Field | Shape | Convention |",
        "|---|---|---|",
        "| `q` | [N,6] | radians, by joint name |",
        "| `q_dot` | [N,6] | radians/s |",
        "| `x` | [N,18] | [sin(q), cos(q), q_dot] |",
        "| `ee_pos_world` | [N,3] | meters, world frame |",
        "| `ee_quat_world` | [N,4] | [w,x,y,z], world frame |",
        "| `ee_rotmat_world` | [N,9] | row-major R, world frame |",
        "| `ee_rot6d_world` | [N,6] | first 2 cols of R |",
        "| `ee_lin_vel_world` | [N,3] | J_pos @ q_dot, world frame |",
        "| `ee_ang_vel_world` | [N,3] | J_rot @ q_dot, world frame |",
        "| `J_world` | [N,6,6] | [J_pos; J_rot], world frame |",
        "| `y` | [N,15] | [ee_pos(3), ee_rot6d(6), lin_vel(3), ang_vel(3)] |",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {output_path}")
