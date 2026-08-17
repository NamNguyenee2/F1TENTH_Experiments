"""
baseline_mpcc.py — Simulate conventional MPCC on the Oschersleben racetrack.

Run:
    python baseline_mpcc.py

Outputs saved to media/figs/:
  - track_trajectory.pdf   — centerline, borders, ARC trajectory
  - inputs.pdf             — ω(t) and a(t)
  - velocity_progress.pdf  — v vs p_t
"""

import os
import sys
import importlib.util
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))
import track
from track import find_progress, p_max

REPO_ROOT = Path(__file__).resolve().parents[2]
_MPCC_CONFIG = REPO_ROOT / "model" / "auto_car" / "mpcc_parameter.yaml"
_CAR_CONFIG = REPO_ROOT / "model" / "auto_car" / "auto_car.py"
_CAR_CONFIG_SPEC = importlib.util.spec_from_file_location("arc_auto_car", _CAR_CONFIG)
car_config = importlib.util.module_from_spec(_CAR_CONFIG_SPEC)
_CAR_CONFIG_SPEC.loader.exec_module(car_config)

with open(_MPCC_CONFIG, "r") as file:
    param = yaml.safe_load(file)

DT = float(param["DT"])
H = int(param["H"])
N_LAPS = int(param["N_LAPS"])
V_MAX = float(param["V_MAX"])
V_MIN = float(param["V_MIN"])
f_phys = car_config.f_phys
from mpcc import solve_op1, mMarkov

FIG_DIR = "media/figs/auto_car_fig"
os.makedirs(FIG_DIR, exist_ok=True)


def simulate_baseline(zeta_p: float = 1.0, v0: float = 0, max_steps: int = 30_000):
    """
    Run baseline MPCC until the ARC completes N_LAPS laps.

    Returns
    -------
    hist : dict with arrays X, Y, phi, v, p_total, omega, a, time
    """
    # Initial state: start at waypoint 0 with heading from track
    Xcen0, Ycen0 = track.centerline_xy(np.array([0.0]))
    phi0    = track.tangent_angle(np.array([0.0]))[0]
    x_state = np.array([float(Xcen0[0]), float(Ycen0[0]), phi0, v0])

    hist = {k: [] for k in ("X", "Y", "phi", "v", "p_total", "omega", "a", "theta_dot", "time")}

    warm_X, warm_U, warm_P, warm_vartheta = None, None, None, None
    lap_count  = 0
    p_prev     = 0.0
    p_total    = 0.0
    t          = 0.0
    last_input = None
    curr_state = np.array([1.])

    print(f"Starting baseline MPCC  |  p_max = {p_max:.2f} m  |  {N_LAPS} laps")

    for step in range(max_steps):
        # ── OP0: find current progress ──────────────────────────────────
        p_bar = find_progress(x_state[0], x_state[1], p_total % p_max)
        # Accumulate total progress (handle lap wrap)
        if p_prev > p_max * 0.9 and p_bar < p_max * 0.1:
            lap_count += 1
            print(f"  Lap {lap_count} completed at t = {t:.2f} s")
        p_total = lap_count * p_max + p_bar

        if lap_count >= N_LAPS:
            print(f"Finished {N_LAPS} laps at t = {t:.2f} s  (step {step})")
            break

        p_prev = p_bar

        cs = mMarkov(curr_state[-1])
        curr_state = np.append(curr_state, cs)

        # ── OP1: solve MPCC ──────────────────────────────────────────────
        X_sol, U_sol, P_sol, S_sol, ok = solve_op1(
            x_state,
            p_total,
            cs,
            warm_X,
            warm_U,
            warm_P,
            warm_vartheta,
        )

        if not ok:
            print(f"OP1 solver failed at step {step}, t = {t:.2f} s. Stopping baseline MPCC.")
            print(
                "Last ARC state: "
                f"X = {float(x_state[0]):.6f}, "
                f"Y = {float(x_state[1]):.6f}, "
                f"phi = {float(x_state[2]):.6f}, "
                f"v = {float(x_state[3]):.6f}, "
                f"p_total = {float(p_total):.6f}"
            )
            if last_input is None:
                print("Last ARC input: none (OP1 failed before any input was applied)")
            else:
                print(
                    "Last ARC input: "
                    f"omega = {last_input['omega']:.6f}, "
                    f"a = {last_input['a']:.6f}, "
                    f"vartheta = {last_input['vartheta']:.6f}, "
                    f"applied_at_t = {last_input['time']:.2f} s"
                )
            break

        # Apply first control input from solution
        u_apply = U_sol[0]
        omega   = float(u_apply[0])
        a       = float(u_apply[1])
        vartheta = float(S_sol[0])

        # Record
        hist["X"].append(float(x_state[0]))
        hist["Y"].append(float(x_state[1]))
        hist["phi"].append(float(x_state[2]))
        hist["v"].append(float(x_state[3]))
        hist["p_total"].append(p_total)
        hist["omega"].append(omega)
        hist["a"].append(a)
        hist["theta_dot"].append(vartheta)
        hist["time"].append(t)

        # ── Propagate dynamics ───────────────────────────────────────────
        x_state = np.array(f_phys(x_state, np.asarray([omega, a]), zeta_p), dtype=float, copy=True)
        x_state[3] = np.clip(x_state[3], V_MIN, V_MAX)
        p_total += vartheta
        t += DT
        last_input = {
            "omega": omega,
            "a": a,
            "vartheta": vartheta,
            "time": hist["time"][-1],
        }

        # Warm start: shift horizon
        warm_X = np.vstack([X_sol[1:], X_sol[-1:]])
        warm_U = np.vstack([U_sol[1:], U_sol[-1:]])
        warm_P = np.concatenate([P_sol[1:], P_sol[-1:]])
        warm_vartheta = np.concatenate([S_sol[1:], S_sol[-1:]])

    else:
        print(f"Max steps reached ({max_steps})")

    return {k: np.array(v) for k, v in hist.items()}


def plot_results(hist):
    # ── Figure 1: Track + trajectory ────────────────────────────────────
    center, left, right = track.border_lines(n_pts=800)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(center[:, 0], center[:, 1], "r--", lw=1.0, label="Centerline")
    ax.plot(left[:, 0],   left[:, 1],   "k-",  lw=1.1, label="Border")
    ax.plot(right[:, 0],  right[:, 1],  "k-",  lw=1.)
    ax.plot(hist["X"],    hist["Y"],    "b-",  lw=0.8, label="ARC trajectory")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Baseline MPCC — Oschersleben")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"track_trajectory_H={H}.pdf"))
    plt.close(fig)

    # ── Figure 2: Inputs ─────────────────────────────────────────────────
    t = hist["time"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, hist["omega"], label="ω (steering rate)")
    axes[0].set_ylabel("ω [rad/s]")
    axes[0].legend()
    axes[1].plot(t, hist["a"], label="a (acceleration)", color="tab:orange")
    axes[1].set_ylabel("a [m/s²]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    fig.suptitle("Control Inputs — Baseline MPCC")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"inputs_H={H}.pdf"))
    plt.close(fig)

    # ── Figure 3: Velocity vs progress ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist["p_total"], hist["v"])
    ax.set_xlabel("Progress p_t [m]")
    ax.set_ylabel("Velocity v [m/s]")
    ax.set_title("Velocity vs Progress — Baseline MPCC")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"velocity_progress_H={H}.pdf"))
    plt.close(fig)

    print(f"Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    hist = simulate_baseline()
    plot_results(hist)
