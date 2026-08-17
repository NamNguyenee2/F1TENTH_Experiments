#!/usr/bin/env python3
"""Inspect UR3e model joints, actuators, and ee_site."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import argparse
import mujoco
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    if args.config:
        from ur3_mujoco.utils.config_loader import load_config
        config = load_config(str(BASE_DIR / args.config))
        model_path = str(BASE_DIR / config['robot']['model_path'])
        site_name = config['robot']['ee_site_name']
    else:
        model_path = str(BASE_DIR / "models" / "scene.xml")
        site_name = "ee_site"

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print("=" * 60)
    print("UR3e Model Inspection")
    print("=" * 60)
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}")

    print("\nJOINTS:")
    all_ok = True
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"  ERROR: '{name}' not found!")
            all_ok = False
            continue
        print(f"  {name:28s}: id={jid}, qpos={model.jnt_qposadr[jid]}, "
              f"qvel={model.jnt_dofadr[jid]}, range=[{model.jnt_range[jid][0]:.3f}, {model.jnt_range[jid][1]:.3f}]")

    print("\nACTUATORS:")
    for i in range(model.nu):
        print(f"  [{i}] {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")

    print("\nEE SITE:")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id >= 0:
        mujoco.mj_forward(model, data)
        print(f"  '{site_name}': id={site_id}, pos={data.site_xpos[site_id]}")
    else:
        print(f"  ERROR: '{site_name}' not found!")
        all_ok = False

    print()
    print("Gate 2 PASS: All joints and ee_site found." if all_ok else "Gate 2 FAIL.")

if __name__ == "__main__":
    main()
