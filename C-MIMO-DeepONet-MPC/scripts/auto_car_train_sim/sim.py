"""Run MPCC with C-MIMO DeepONet predictions and PANOC or IPOPT."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from baseline.auto_car_deeponet.script_jong2025 import load_bundle as load_jong2025_bundle
from baseline.auto_car_deeponet.script_nghiem2023 import load_bundle as load_nghiem2023_bundle
# from deeponet_panoc.ipopt import make_deeponet_ipopt_solver
# from src.deeponet_panoc.mpcc import (
#     DEEPONET_MODELS,
#     MPCCPANOCConfig,
#     make_deeponet_panoc_solver,
# )
# from src.deeponet_panoc.panoc import PANOCConfig
# from src.deeponet_panoc.track_jax import build_track_data
from src.deeponet import load_bundle as load_CMIMO_bundle
from src.deeponet_panoc import (
    make_deeponet_ipopt_solver,
    DEEPONET_MODELS,
    MPCCPANOCConfig,
    make_deeponet_panoc_solver,
    PANOCConfig,
    build_track_data,
)
from model.auto_car.auto_car import f_phys
from baseline.auto_car_dynamics_ipopt import track

_MPCC_CONFIG = REPO_ROOT / "model" / "auto_car" / "mpcc_parameter.yaml"

with open(_MPCC_CONFIG, "r") as file:
    param = yaml.safe_load(file)

DT = float(param["DT"])
H = int(param["H"])
N_LAPS = int(param["N_LAPS"])
V_MAX = float(param["V_MAX"])
V_MIN = float(param["V_MIN"])


FIG_DIR = REPO_ROOT / "media" / "figs" / "auto_car_fig"

DEFAULT_MODEL_PATHS = {
    "CATR_MIMO_deeponet": REPO_ROOT / "model" / "auto_car_learning" / "CATR_MIMO_DeepONet_AC_PP_50000_07271410_p=20_b=1_e=32_br=96_tr=96.npz",
    "jong2025": REPO_ROOT / "model" / "auto_car_learning" / "DeepONet_jong2025_AC_PP_50000_07271410_p=20_br=96_tr=96.npz",
    "nghiem2023": REPO_ROOT / "model" / "auto_car_learning" / "DeepONet_nghiem2023_AC_PP_50000_07271410_p=20_h=48_alpha=20.npz",
}

MODEL_LOADERS = {
    "CATR_MIMO_deeponet": load_CMIMO_bundle,
    "jong2025": load_jong2025_bundle,
    "nghiem2023": load_nghiem2023_bundle,
}


def simulate(
    bundle,
    track_data,
    dynamics_model: str = "CATR_MIMO_deeponet",
    solver: str = "panoc",
    max_steps: int = 300,
    n_laps: int = N_LAPS,
    v0: float = V_MAX / 2,
    panoc_max_iter: int = 100,
    panoc_tol: float = 1e-3,
    gamma: float = 1e-2,
    ipopt_max_iter: int = 1000,
    ipopt_tol: float = 1e-4,
):
    x0_cen, y0_cen = track.centerline_xy(np.array([0.0]))
    phi0 = float(track.tangent_angle(np.array([0.0]))[0])
    x_state = np.array([float(x0_cen[0]), float(y0_cen[0]), phi0, float(v0)], dtype=np.float32)

    hist = {k: [] for k in ("X", "Y", "phi", "v", "p_total", "omega", "a", "theta_dot", "time", "residual", "iters")}
    warm_u = None
    lap_count = 0
    p_prev = 0.0
    p_total = 0.0
    t = 0.0

    config = MPCCPANOCConfig(
        dynamics_model=dynamics_model,
        panoc=PANOCConfig(max_iter=panoc_max_iter, tol=panoc_tol, gamma=gamma, lbfgs_memory=10)
    )
    if solver == "ipopt" and dynamics_model != "kinematic":
        raise ValueError("IPOPT currently supports only the kinematic model; use --solver panoc for DeepONet models.")

    if solver == "panoc":
        solve_mpc = make_deeponet_panoc_solver(bundle=bundle, track_data=track_data, config=config)
    elif solver == "ipopt":
        solve_mpc = make_deeponet_ipopt_solver(
            bundle=bundle,
            config=config,
            max_iter=ipopt_max_iter,
            tol=ipopt_tol,
        )
    else:
        raise ValueError(f"Unknown solver={solver!r}")

    run_label = _run_label(dynamics_model, solver)
    print(f"Starting {run_label} MPCC | p_max = {track.p_max:.2f} m | target laps = {n_laps}")
    for step in range(max_steps):
        p_bar = track.find_progress(float(x_state[0]), float(x_state[1]), p_total % track.p_max)
        if p_prev > track.p_max * 0.9 and p_bar < track.p_max * 0.1:
            lap_count += 1
            print(f"  Lap {lap_count} completed at t = {t:.2f} s")
        p_total = lap_count * track.p_max + p_bar
        if lap_count >= n_laps:
            break
        p_prev = p_bar

        sol     = solve_mpc(x0=x_state, p0=p_total, warm_u=warm_u)
        if solver == "ipopt" and not sol["converged"]:
            print(f"IPOPT solver failed at step {step}, t = {t:.2f} s. Stopping {_run_label(dynamics_model, solver)} MPCC.")
            break
        u_apply = sol["U"][0]
        omega   = float(u_apply[0])
        accel   = float(u_apply[1])
        vartheta = float(sol["S"][0])

        hist["X"].append(float(x_state[0]))
        hist["Y"].append(float(x_state[1]))
        hist["phi"].append(float(x_state[2]))
        hist["v"].append(float(x_state[3]))
        hist["p_total"].append(float(p_total))
        hist["omega"].append(omega)
        hist["a"].append(accel)
        hist["theta_dot"].append(vartheta)
        hist["time"].append(t)
        hist["residual"].append(sol["residual_norm"])
        hist["iters"].append(sol["iterations"])

        x_state = np.array(f_phys(x_state, np.asarray([omega, accel], dtype=np.float32)), dtype=np.float32, copy=True)
        x_state[3] = np.clip(x_state[3], V_MIN, V_MAX)
        p_total += vartheta
        t += DT
        if solver == "ipopt":
            warm_u = {
                "X": np.vstack([sol["X"][1:], sol["X"][-1:]]),
                "U": np.vstack([sol["U"][1:], sol["U"][-1:]]),
                "P": np.concatenate([sol["P"][1:], sol["P"][-1:]]),
                "S": np.concatenate([sol["S"][1:], sol["S"][-1:]]),
            }
        else:
            warm_u = np.vstack([sol["U"][1:], sol["U"][-1:]])

        if step % 10 == 0:
            print(
                f"step={step:04d} t={t:.2f} p={p_total:.2f} v={x_state[3]:.2f} "
                f"u=({omega:.3f},{accel:.3f}) residual={sol['residual_norm']:.3e} "
                f"iters={sol['iterations']}"
            )

    return {k: np.asarray(v) for k, v in hist.items()}


def _run_label(dynamics_model: str, solver: str) -> str:
    model_labels = {
        "CATR_MIMO_deeponet": "C-MIMO DeepONet (proposed)",
        "jong2025": "Jong2025 DeepONet",
        "nghiem2023": "Nghiem2023 DeepONet",
        "kinematic": "Kinematic",
    }
    model_label = model_labels[dynamics_model]
    return f"{model_label}+{solver.upper()}"


def _run_slug(dynamics_model: str, solver: str) -> str:
    return f"{dynamics_model}_{solver}"


def plot_results(hist, dynamics_model: str, solver: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    run_label = _run_label(dynamics_model, solver)
    run_slug  = _run_slug(dynamics_model, solver)

    # Tracking: track map and ARC closed-loop trajectory.
    center, left, right = track.border_lines(n_pts=800)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(center[:, 0], center[:, 1], "r--", lw=1.0, label="Centerline")
    ax.plot(left[:, 0], left[:, 1], "k-", lw=1.1, label="Border")
    ax.plot(right[:, 0], right[:, 1], "k-", lw=1.0)
    ax.plot(hist["X"], hist["Y"], "b-", lw=0.8, label="ARC trajectory")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Tracking — {run_label} MPCC, H={H}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{run_slug}_tracking_H={H}.pdf")
    plt.close(fig)

    # Velocity: expose both progress-relative and time-relative behavior.
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    axes[0].plot(hist["time"], hist["v"], color="tab:green")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("v [m/s]")
    axes[1].plot(hist["p_total"], hist["v"], color="tab:green")
    axes[1].set_xlabel("progress p [m]")
    axes[1].set_ylabel("v [m/s]")
    fig.suptitle(f"Velocity — {run_label} MPCC")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{run_slug}_velocity_H={H}.pdf")
    plt.close(fig)

    # Inputs: applied ARC controls.
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(hist["time"], hist["omega"], label="omega")
    axes[0].set_ylabel("omega [rad/s]")
    axes[0].legend()
    axes[1].plot(hist["time"], hist["a"], color="tab:orange", label="a")
    axes[1].set_ylabel("a [m/s^2]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend()
    fig.suptitle(f"Inputs — {run_label} MPCC")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{run_slug}_inputs_H={H}.pdf")
    plt.close(fig)
    print(f"Figures saved to {FIG_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run JAX DeepONet MPCC with PANOC or CasADi/IPOPT.")
    parser.add_argument(
        "--solver",
        choices=("panoc", "ipopt"),
        default="panoc",
        help="NLP solver used for the MPCC.",
    )
    parser.add_argument(
        "--model",
        "--dynamics-model",
        dest="dynamics_model",
        choices=(*DEEPONET_MODELS, "kinematic"),
        default="CATR_MIMO_deeponet",
        help="Prediction model used inside MPCC (default: CATR_MIMO_deeponet).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional checkpoint override for the selected DeepONet model.",
    )
    parser.add_argument("--max-steps",      type=int,   default=1000)
    parser.add_argument("--n-laps",         type=int,   default=N_LAPS)
    parser.add_argument("--panoc-max-iter", type=int,   default=100)
    parser.add_argument("--panoc-tol",      type=float, default=1e-2)
    parser.add_argument("--panoc-gamma",    type=float, default=1e-2)
    parser.add_argument("--ipopt-max-iter", type=int,   default=1000)
    parser.add_argument("--ipopt-tol",      type=float, default=1e-2)

    return parser.parse_args()


def main():
    args = parse_args()

    bundle = None
    if args.dynamics_model in DEEPONET_MODELS:
        model_path = args.model_path or DEFAULT_MODEL_PATHS[args.dynamics_model]
        if not model_path.is_file():
            raise FileNotFoundError(f"Checkpoint for {args.dynamics_model!r} not found: {model_path}")
        bundle = MODEL_LOADERS[args.dynamics_model](model_path)
        print(f"Loaded {args.dynamics_model} model from {model_path}")
    elif args.model_path is not None:
        raise ValueError("--model-path cannot be used with --model kinematic")

    track_data = build_track_data()

    hist = simulate(
        bundle          =   bundle,
        track_data      =   track_data,
        dynamics_model  =   args.dynamics_model,
        solver          =   args.solver,
        max_steps       =   args.max_steps,
        n_laps          =   args.n_laps,
        panoc_max_iter  =   args.panoc_max_iter,
        panoc_tol       =   args.panoc_tol,
        gamma           =   args.panoc_gamma,
        ipopt_max_iter  =   args.ipopt_max_iter,
        ipopt_tol       =   args.ipopt_tol,
    )

    if len(hist["time"]) == 0:
        raise RuntimeError(f"{_run_label(args.dynamics_model, args.solver)} simulation produced no steps")

    plot_results(hist, args.dynamics_model, args.solver)
    
    print(
        f"Completed {_run_label(args.dynamics_model, args.solver)} run: "
        f"steps={len(hist['time'])}, "
        f"final_progress={hist['p_total'][-1]:.3f}, "
        f"median_residual={np.median(hist['residual']):.3e}"
    )


if __name__ == "__main__":
    main()
