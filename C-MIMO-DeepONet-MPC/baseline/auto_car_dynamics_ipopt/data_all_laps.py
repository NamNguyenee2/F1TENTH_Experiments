"""
Each sample is (u, x).
The states are sampled near the Oschersleben centerline and labels are solved with CasADi.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
from pathlib import Path
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import random
import yaml


import track
from mpcc import solve_op1_horizon
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MPCC_CONFIG = _REPO_ROOT / "model" / "auto_car" / "mpcc_parameter.yaml"
_CAR_CONFIG = _REPO_ROOT / "model" / "auto_car" / "auto_car.py"
_CAR_CONFIG_SPEC = importlib.util.spec_from_file_location("AC_auto_car", _CAR_CONFIG)
car_config = importlib.util.module_from_spec(_CAR_CONFIG_SPEC)
_CAR_CONFIG_SPEC.loader.exec_module(car_config)

with open(_MPCC_CONFIG, "r") as file:
    param = yaml.safe_load(file)

DATA_DIR = param["DATA_DIR"]
DELTA_MAX = float(param["DELTA_MAX"])
DT = float(param["DT"])
H = int(param["H"])
N_DATA = int(param["N_DATA"])
ZETA_P = float(param["ZETA_P"])
V_MAX = float(param["V_MAX"])
V_MIN = float(param["V_MIN"])
Q_V = float(param["Q_V"])
f_phys = car_config.f_phys

FIG_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "training"))

v_int = V_MAX/2


def build_v_max_schedule(n_data: int) -> np.ndarray:
    steps = np.random.uniform(low = -0.2, high = 0.2, size = n_data-1) 

    cum    = np.zeros(len(steps)+1)
    cum[0] = v_int

    for i in range(len(steps)):
        temp = cum[i] + steps[i]
        if temp >= V_MAX + 1.:
            cum[i+1] = V_MAX + 1.
        elif temp <= V_MIN : 
            cum[i+1] = V_MIN
        else:
            cum[i+1] = temp

    return np.clip(cum, V_MIN, V_MAX)


def sample_state(rng: np.random.Generator, v_max: float = V_MAX) -> np.ndarray:
    """Sample 4D state [X, Y, phi, v] near the reference path."""
    p             = 0.
    xc, yc        = track.centerline_xy(np.array([p]))
    phi_ref       = float(track.tangent_angle(np.array([p]))[0])
    lateral       = rng.uniform(-DELTA_MAX * 0.1, DELTA_MAX * 0.1)
    heading_error = rng.uniform(-0.1, 0.1)
    nx = -np.sin(phi_ref)
    ny =  np.cos(phi_ref)
    x = np.array(
        [
            float(xc[0] + lateral * nx),
            float(yc[0] + lateral * ny),
            phi_ref + heading_error,
            v_int,
        ],
        dtype=np.float64,
    )
    return x


def progress_delta(p_now: float, p_prev: float) -> float:
    """Forward progress increment on a closed track, robust to lap wrap."""
    delta = float(p_now - p_prev)
    if delta < -0.5 * track.p_max:
        delta += track.p_max
    elif delta > 0.5 * track.p_max:
        delta -= track.p_max
    return max(delta, 0.0)


def plot_collected_trajectory(xs, lap_sample_counts, output_dir: str = FIG_DIR):
    """Plot the collected AC data trajectory on the Oschersleben track."""
    os.makedirs(output_dir, exist_ok=True)
    states = np.asarray(xs, dtype=float)
    if states.size == 0:
        return None

    center, left, right = track.border_lines(n_pts=800)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(center[:, 0], center[:, 1], "r--", lw=1.0, label="Centerline")
    ax.plot(left[:, 0], left[:, 1], "k-", lw=1.1, label="Border")
    ax.plot(right[:, 0], right[:, 1], "k-", lw=1.0)
    ax.plot(states[:, 0], states[:, 1], "b-", lw=0.8, label="Collected AC trajectory")

    start = 0
    for lap_idx, count in enumerate(lap_sample_counts):
        if count <= 0:
            continue
        idx = min(start, len(states) - 1)
        ax.plot(states[idx, 0], states[idx, 1], "o", ms=3.5, label="Lap start" if lap_idx == 0 else None)
        start += int(count)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Collected AC Trajectory")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()

    path = os.path.join(output_dir, f"collect_trajectory.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def collect_all_laps(start_state: np.ndarray, label_horizon: int, n_data: int = N_DATA):
    """Collect n_data samples from one continuous  trajectory ross lap wraps."""

    x_state = np.asarray(start_state, dtype=np.float64).copy()
    v_max_schedule = build_v_max_schedule(n_data)

    p_prev  = track.find_progress(float(x_state[0]), float(x_state[1]), 0.0, n_seACh=1, n_grid=401)

    lap_progress = 0.0
    next_lap_progress = track.p_max
    current_lap_count = 0
    warm_X, warm_U, warm_P, warm_vartheta = None, None, None, None

    xs, u0, labels = [], [], []
    lap_sample_counts = []

    for i in range(n_data):
        current_v_max = float(v_max_schedule[i])

        p_now = track.find_progress(float(x_state[0]), float(x_state[1]), p_prev, n_seACh=5, n_grid=101)
        lap_progress += progress_delta(p_now, p_prev)
        p_prev = p_now

        while lap_progress >= next_lap_progress and current_lap_count > 0:
            lap_sample_counts.append(current_lap_count)
            current_lap_count = 0
            next_lap_progress += track.p_max

        X_sol, U_sol, P_sol, S_sol, value, status = solve_op1_horizon(
            x0=x_state,
            horizon=label_horizon,
            warm_X=warm_X,
            warm_U=warm_U,
            warm_P=warm_P,
            warm_vartheta=warm_vartheta,
            v_max=current_v_max,
        )

        if i%100 ==0:
            print(f"Collected {i} data after {len(lap_sample_counts)} laps | V_MAX = {current_v_max:.2f} | V_sol = {X_sol[1,3]}")

        if not status or not np.isfinite(value):
            print(f"Infeasible or Undefined v_max = {v_max_schedule[i]}")
            return None

        xs.append(x_state.astype(np.float32))
        U_sol[0][1] += random.uniform(-2.,  0.)
        u0.append(U_sol[0].astype(np.float32))
        labels.append(float(value))
        current_lap_count += 1

        omega, a = U_sol[0]
 

        x_state = np.array(f_phys(x_state, np.asarray([omega, a]), ZETA_P), dtype=np.float64, copy=True)

        warm_X = np.vstack([X_sol[1:], X_sol[-1:]])
        warm_U = np.vstack([U_sol[1:], U_sol[-1:]])
        warm_P = np.concatenate([P_sol[1:], P_sol[-1:]])
        warm_vartheta = np.concatenate([S_sol[1:], S_sol[-1:]])

    if current_lap_count > 0:
        lap_sample_counts.append(current_lap_count)

    return xs, u0, labels, lap_sample_counts


def generate_dataset(
    n_data: int = N_DATA,
    seed: int = 0,
    output_dir: str = DATA_DIR,
    max_starting_points: int = 10,
):

    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    random.seed(seed)

    xs, u0, labels = [], [], []
    lap_sample_counts = []

    start_state = sample_state(rng, V_MAX)

    lap_data  = collect_all_laps(start_state, H, n_data)

    xs, u0, labels, lap_sample_counts = lap_data

    print( f"{len(labels)} samples across {len(lap_sample_counts)} lap segments")


    if not labels:
        raise RuntimeError("No feasible OP1 labels were generated")
    if len(labels) < n_data:
        raise RuntimeError(
            f"Only generated {len(labels)} feasible labels after {attempts} attempts; "
            f"requested {n_data}"
        )

    stamp   = datetime.now().strftime("%m%d%H%M")
    count   = len(labels)
    path    = os.path.join(output_dir, f"AC_MPCC_{count}_{stamp}.npz")
    np.savez(
        path,
        x = np.asarray(xs, dtype=np.float32),
        u = np.asarray(u0, dtype=np.float32),
        sample_time = np.asarray(DT, dtype=np.float32),
        lap_sample_counts = np.asarray(lap_sample_counts, dtype=np.int32),
    )
    print(f"Saved {count} samples to {path}")
    fig_path = plot_collected_trajectory(xs, lap_sample_counts)
    if fig_path is not None:
        print(f"Saved collected trajectory figure to {fig_path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CTG training data.")
    parser.add_argument("--n-data", type=int, default=N_DATA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=DATA_DIR)
    parser.add_argument("--max-starting-points", type=int, default=10_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args.n_data, args.seed, args.output_dir, args.max_starting_points)
