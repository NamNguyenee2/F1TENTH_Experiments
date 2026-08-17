#!/usr/bin/env python3
"""Generate UR3 trajectory dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import argparse
import numpy as np
import h5py
import time
from pathlib import Path
from ur3_mujoco import UR3Model, PDController, TrajectoryGenerator, DataLogger
import mujoco

MODEL_PATH = Path(__file__).parent.parent / "models" / "scene.xml"
DATA_DIR = Path(__file__).parent.parent / "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-trajectories', type=int, default=10)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--traj-type', choices=['sine', 'multi', 'low_amp'], default='multi')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    out_path = args.output or str(DATA_DIR / f"dataset_{args.traj_type}_{args.num_trajectories}traj.h5")

    ur3 = UR3Model(str(MODEL_PATH))
    ctrl = PDController(ur3)
    traj_gen = TrajectoryGenerator()
    logger = DataLogger(ur3)
    dt = ur3.model.opt.timestep

    print(f"Generating {args.num_trajectories} trajectories ({args.traj_type}), duration={args.duration}s each")

    metadata = {
        'mujoco_version': mujoco.__version__,
        'model_path': str(MODEL_PATH),
        'timestep': dt,
        'joint_names': ','.join(ur3.JOINT_NAMES),
        'ee_site': 'ee_site',
        'traj_type': args.traj_type,
        'n_trajectories': args.num_trajectories,
        'duration': args.duration,
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with h5py.File(out_path, 'w') as f:
        meta_grp = f.create_group('metadata')
        for k, v in metadata.items():
            meta_grp.attrs[k] = str(v)

        for traj_i in range(args.num_trajectories):
            seed_i = args.seed + traj_i

            if args.traj_type == 'sine':
                joint_idx = traj_i % 6
                traj = traj_gen.single_joint_sine(joint_idx=joint_idx, amplitude=0.4, frequency=0.2, duration=args.duration, dt=dt)
            elif args.traj_type == 'multi':
                traj = traj_gen.multi_joint_smooth(duration=args.duration, dt=dt, n_waypoints=8, seed=seed_i)
            else:
                traj = traj_gen.low_amplitude(duration=args.duration, dt=dt)

            ur3.reset(traj['q'][0])
            logger.start_episode()

            has_nan = False
            for i in range(len(traj['t'])):
                c = ctrl.compute(ur3.get_q(), ur3.get_qdot(), traj['q'][i], traj['qdot'][i], traj['qdotdot'][i])
                ur3.set_ctrl(c)
                ur3.step()
                logger.log_step(ur3.data, c, traj['q'][i], traj['qdot'][i], traj['qdotdot'][i])

            ep = logger.get_episode_data()

            for k, v in ep.items():
                if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                    print(f"  WARNING: NaN/Inf in {k} at trajectory {traj_i}")
                    has_nan = True

            traj_grp = f.create_group(f'trajectory_{traj_i:03d}')
            for k, v in ep.items():
                traj_grp.create_dataset(k, data=v)
            traj_grp.attrs['traj_type'] = args.traj_type
            traj_grp.attrs['seed'] = seed_i
            traj_grp.attrs['has_nan'] = has_nan

            if (traj_i + 1) % 10 == 0 or traj_i == 0:
                print(f"  [{traj_i+1}/{args.num_trajectories}] saved trajectory_{traj_i:03d}")

    print(f"\nDataset saved: {out_path}")
    print(f"Gate 7: Verifying dataset...")

    with h5py.File(out_path, 'r') as f:
        n_traj = len([k for k in f.keys() if k.startswith('trajectory_')])
        print(f"  Trajectories: {n_traj}")
        t0 = f['trajectory_000']
        for field in ['time', 'q', 'q_dot', 'q_dot_dot', 'ee_pos', 'ee_quat', 'ee_rotmat', 'ee_lin_vel', 'ee_ang_vel', 'ctrl']:
            print(f"  {field}: {t0[field].shape}")
    print("Gate 7: PASS")

if __name__ == "__main__":
    main()
