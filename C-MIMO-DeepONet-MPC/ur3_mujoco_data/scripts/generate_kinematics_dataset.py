#!/usr/bin/env python3
"""Generate UR3e kinematics dataset for supervised learning."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import numpy as np
from pathlib import Path

from ur3_mujoco.utils.config_loader import load_config
from ur3_mujoco.utils.random_utils import sample_q, sample_qdot
from ur3_mujoco.sim.mujoco_env import MujocoEnv
from ur3_mujoco.robot.ur3_robot import UR3Robot
from ur3_mujoco.dataset.kinematics_sampler import sample_one
from ur3_mujoco.dataset.kinematics_writer import write_dataset
from ur3_mujoco.trajectory import single_joint_sine, multi_joint_sine, random_spline

BASE_DIR = Path(__file__).parent.parent


def generate_random_samples(robot, config, seed, n_q, n_qdot_per_q, label="",
                            vel_limits_key='q_velocity_limits'):
    jl = np.array(config['robot']['joint_limits'])
    vl = config['robot'][vel_limits_key]
    frac = config['robot'].get('safe_joint_fraction', 0.7)
    rng = np.random.default_rng(seed)

    N = n_q * n_qdot_per_q
    print(f"  Sampling {N} points ({label}, seed={seed})...")
    print(f"  q_dot limits: {['[%.1f,%.1f]'%(v[0],v[1]) for v in vl]}")
    results = []
    for qi in range(n_q):
        q = sample_q(rng, jl, frac)
        for _ in range(n_qdot_per_q):
            qdot = sample_qdot(rng, vl)
            results.append(sample_one(robot, q, qdot))
        if (qi + 1) % max(1, n_q // 5) == 0:
            print(f"    q {qi+1}/{n_q} done")
    return {k: np.stack([r[k] for r in results]) for k in results[0]}


def run_trajectory_forward(robot, traj: dict) -> dict:
    """Run each (q, qdot) through mj_forward and collect supervised labels."""
    q_arr = traj['q']
    qdot_arr = traj['qdot']
    qdotdot_arr = traj.get('qdotdot', np.zeros_like(q_arr))
    t_arr = traj['t']
    N = len(t_arr)

    rows = [sample_one(robot, q_arr[i], qdot_arr[i]) for i in range(N)]
    result = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    result['time'] = t_arr
    result['q_dot_dot'] = qdotdot_arr
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config = load_config(str(BASE_DIR / args.config))
    model_path = str(BASE_DIR / config['robot']['model_path'])

    print(f"Loading model: {model_path}")
    env = MujocoEnv(model_path)
    robot = UR3Robot(env, ee_site_name=config['robot']['ee_site_name'])
    print(f"Stage: {config['dataset']['stage']}")

    dc = config['dataset']
    rc = dc['random']
    tc = dc.get('trajectories', {})
    dt = dc['simulation']['timestep']

    n_q_total = rc['n_q_samples']
    n_qdot = rc['n_qdot_per_q']
    val_frac = rc.get('val_fraction', 0.1)
    test_frac = rc.get('test_fraction', 0.1)

    n_val_q  = max(1, int(n_q_total * val_frac))
    n_test_q = max(1, int(n_q_total * test_frac))
    n_train_q = n_q_total - n_val_q - n_test_q

    # ── 1. Random samples (independent seeds per split) ─────────────
    print("\n[1/5] Random supervised samples (normal speed)")
    train_data = generate_random_samples(robot, config, rc['seed_train'],  n_train_q, n_qdot, "train")
    val_data   = generate_random_samples(robot, config, rc['seed_val'],    n_val_q,   n_qdot, "val")
    test_data  = generate_random_samples(robot, config, rc['seed_test'],   n_test_q,  n_qdot, "test_random")
    print(f"  train={len(train_data['q'])}, val={len(val_data['q'])}, test_random={len(test_data['q'])}")

    # ── 1b. Stress-test split (higher q_dot range) ──────────────────
    stress_data = None
    if 'q_velocity_limits_stress' in config['robot']:
        print("\n[1b] Stress-test samples (high speed)")
        n_stress_q = max(1, int(n_test_q * 0.5))   # half the size of test split
        stress_data = generate_random_samples(
            robot, config, rc['seed_test'] + 999, n_stress_q, n_qdot,
            "test_stress", vel_limits_key='q_velocity_limits_stress')
        print(f"  test_stress={len(stress_data['q'])}")

    # ── 2. Single-joint sine trajectories ───────────────────────────
    print("\n[2/5] Single-joint sine trajectories")
    sjs_cfg = tc.get('single_joint_sine', {})
    amp_rad = np.deg2rad(sjs_cfg.get('amplitude_deg', 20.0))
    freq    = sjs_cfg.get('frequency', 0.1)
    dur     = sjs_cfg.get('duration', 10.0)
    sjs_trajs = []
    for ji in range(6):
        traj = single_joint_sine.generate(ji, amp_rad, freq, dur, dt)
        result = run_trajectory_forward(robot, traj)
        result['trajectory_id'] = ji
        sjs_trajs.append(result)
        print(f"  joint {ji} done ({len(traj['t'])} steps)")

    # ── 3. Multi-joint sine trajectories ────────────────────────────
    print("\n[3/5] Multi-joint sine trajectories")
    mjs_cfg = tc.get('multi_joint_sine', {})
    n_mjs = mjs_cfg.get('n_trajectories', 5)
    rng_mjs = np.random.default_rng(mjs_cfg.get('seed', 4000))
    mjs_trajs = []
    for i in range(n_mjs):
        amp_range  = mjs_cfg.get('amplitude_deg_range', [10.0, 30.0])
        freq_range = mjs_cfg.get('frequency_range', [0.05, 0.2])
        amps   = np.deg2rad(rng_mjs.uniform(amp_range[0],  amp_range[1],  6))
        freqs  = rng_mjs.uniform(freq_range[0], freq_range[1], 6)
        phases = rng_mjs.uniform(0, 2 * np.pi, 6)
        traj = multi_joint_sine.generate(amps, freqs, phases, mjs_cfg.get('duration', 20.0), dt)
        result = run_trajectory_forward(robot, traj)
        result['trajectory_id'] = i
        mjs_trajs.append(result)
    print(f"  {n_mjs} trajectories done")

    # ── 4. Random spline trajectories (test_trajectory split) ───────
    print("\n[4/5] Random spline trajectories")
    rsp_cfg = tc.get('random_spline', {})
    n_rsp = rsp_cfg.get('n_trajectories', 10)
    rng_rsp = np.random.default_rng(rsp_cfg.get('seed_test', 5000))
    spline_trajs = []
    for i in range(n_rsp):
        traj = random_spline.generate(
            rng_rsp,
            n_waypoints=rsp_cfg.get('n_waypoints', 6),
            duration=rsp_cfg.get('duration', 10.0),
            dt=dt,
            joint_range_fraction=rsp_cfg.get('joint_range_fraction', 0.6),
        )
        result = run_trajectory_forward(robot, traj)
        result['trajectory_id'] = i
        spline_trajs.append(result)
    print(f"  {n_rsp} trajectories done")

    # Concatenate spline trajectories into test_trajectory split
    traj_split_keys = [k for k in spline_trajs[0] if k not in ('trajectory_id', 'time', 'q_dot_dot')]
    test_traj_data = {k: np.concatenate([t[k] for t in spline_trajs], axis=0) for k in traj_split_keys}

    # ── 5. Summary ──────────────────────────────────────────────────
    print("\n[5/5] Writing dataset")
    splits = {
        "train":           train_data,
        "val":             val_data,
        "test_random":     test_data,
        "test_trajectory": test_traj_data,
    }
    if stress_data is not None:
        splits["test_stress"] = stress_data
    traj_groups = {
        "debug_trajectories": {
            "single_joint_sine": sjs_trajs,
            "multi_joint_sine":  mjs_trajs,
        }
    }

    write_dataset(args.output, splits, traj_groups, config, model_path)
    print(f"\nDone: {args.output}")


if __name__ == "__main__":
    main()
