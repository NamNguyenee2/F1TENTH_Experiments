"""MPCC objective using C-MIMO DeepONet predictions and JAX PANOC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.deeponet import DeepONetBundle as DeepONetBundle_mine
from src.deeponet import predict_states as predict_states_mine
from src.deeponet import predict_states_u_jacobian as predict_states_u_jacobian_mine

from baseline.auto_car_deeponet.script_nghiem2023 import DeepONetBundle as DeepONetBundle_nghiem2023
from baseline.auto_car_deeponet.script_nghiem2023 import predict_states as predict_states_root_nghiem2023
from baseline.auto_car_deeponet.script_nghiem2023 import predict_states_u_jacobian as predict_states_u_jacobian_nghiem2023

from baseline.auto_car_deeponet.script_jong2025 import DeepONetBundle as DeepONetBundle_jong2025
from baseline.auto_car_deeponet.script_jong2025 import predict_states as predict_states_jong2025
from baseline.auto_car_deeponet.script_jong2025 import predict_states_u_jacobian as predict_states_u_jacobian_jong2025

from src.deeponet_panoc.panoc import PANOCConfig, make_panoc_solver
from src.deeponet_panoc.track_jax import TrackData, eval_track
from model.auto_car.auto_car import f_phys


_MPCC_CONFIG = REPO_ROOT / "model" / "auto_car" / "mpcc_parameter.yaml"

with open(_MPCC_CONFIG, "r") as file:
    param = yaml.safe_load(file)

A_MAX = float(param["A_MAX"])
A_MIN = float(param["A_MIN"])
DELTA_MAX = float(param["DELTA_MAX"])
DT = float(param["DT"])
H = int(param["H"])
OMEGA_MAX = float(param["OMEGA_MAX"])
OMEGA_MIN = float(param["OMEGA_MIN"])
Q_A = float(param["Q_A"])
Q_C = float(param["Q_C"])
Q_L = float(param["Q_L"])
Q_OMEGA = float(param["Q_OMEGA"])
Q_V = float(param["Q_V"])
V_MAX = float(param["V_MAX"])
V_MIN = float(param["V_MIN"])
ZETA_P = float(param["ZETA_P"])


DynamicsModel = Literal["CATR_MIMO_deeponet", "nghiem2023", "jong2025", "kinematic"]
DeepONetBundle = (
    DeepONetBundle_nghiem2023
    | DeepONetBundle_jong2025
    | DeepONetBundle_mine
)
DEEPONET_MODELS = ("CATR_MIMO_deeponet", "nghiem2023", "jong2025")


@dataclass(frozen=True)
class MPCCPANOCConfig:
    state_penalty:    float = 1000.
    velocity_penalty: float = 1000.
    terminal_weight:  float = 1.
    dynamics_model: DynamicsModel = "CATR_MIMO_deeponet"
    panoc: PANOCConfig = PANOCConfig()


def penalty_func(x):
    return jax.nn.relu(x)


def predict_kinematic_states(x0: jnp.ndarray, u_seq: jnp.ndarray) -> jnp.ndarray:
    """Roll out the baseline auto_car_dynamics_ipopt kinematic ARC model."""

    def step(x_prev, u_i):
        x_next = jnp.asarray(f_phys(x_prev, u_i, ZETA_P), dtype=jnp.float32)
        return x_next, x_next

    _, x_seq = jax.lax.scan(step, jnp.asarray(x0, dtype=jnp.float32), jnp.asarray(u_seq, dtype=jnp.float32))
    return x_seq


def _predict_states_for_model(
    bundle: DeepONetBundle | None,
    dynamics_model: DynamicsModel,
    x0: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> jnp.ndarray:
    if dynamics_model == "kinematic":
        return predict_kinematic_states(x0, u_seq)

    if bundle is None:
        raise ValueError(f"A DeepONet bundle is required for dynamics_model={dynamics_model!r}")

    if dynamics_model == "nghiem2023":
        return predict_states_root_nghiem2023(bundle, x0, u_seq)

    if dynamics_model == "jong2025":
        return predict_states_jong2025(bundle, x0, u_seq)

    if dynamics_model == "CATR_MIMO_deeponet":
        return predict_states_mine(bundle, x0, u_seq)

    raise ValueError(f"Unknown dynamics_model={dynamics_model!r}")


def _predict_states_u_jacobian_for_model(
    bundle: DeepONetBundle,
    x0: jnp.ndarray,
    u_seq: jnp.ndarray,
    dynamics_model: DynamicsModel,
) -> jnp.ndarray:
    if dynamics_model == "nghiem2023":
        return predict_states_u_jacobian_nghiem2023(bundle, x0, u_seq)

    if dynamics_model == "jong2025":
        return predict_states_u_jacobian_jong2025(bundle, x0, u_seq)

    if dynamics_model == "CATR_MIMO_deeponet":
        return predict_states_u_jacobian_mine(bundle, x0, u_seq)

    raise ValueError(f"No DeepONet Jacobian is defined for dynamics_model={dynamics_model!r}")


def _mpcc_cost_from_prediction(
    x_pred: jnp.ndarray,
    u_flat: jnp.ndarray,
    p0: jnp.ndarray,
    track_data: TrackData,
    config: MPCCPANOCConfig,
    horizon: int,
    n_u: int,
) -> jnp.ndarray:
    u_seq = u_flat.reshape((horizon, n_u))
    v_pred = x_pred[:, 3]
    progress_step = DT * jnp.clip(v_pred, 0.0, V_MAX)
    p_seq = p0 + jnp.cumsum(progress_step)
    x_ref, y_ref, phi_ref = eval_track(track_data, p_seq)

    dx = x_pred[:, 0] - x_ref
    dy = x_pred[:, 1] - y_ref
    sin_phi = jnp.sin(phi_ref)
    cos_phi = jnp.cos(phi_ref)
    e_cot = sin_phi * dx - cos_phi * dy
    e_lag = -cos_phi * dx - sin_phi * dy

    omega = u_seq[:, 0]
    accel = u_seq[:, 1]
    stage = (
        Q_C * e_cot**2
        + Q_L * e_lag**2
        + Q_OMEGA * omega**2
        + Q_A * accel**2
        - Q_V * progress_step
    )

    dist_violation = penalty_func(dx**2 + dy**2 - DELTA_MAX**2)
    v_low = penalty_func(V_MIN - v_pred)
    v_high = penalty_func(v_pred - V_MAX)
    penalties = (config.state_penalty * dist_violation**2 + config.velocity_penalty * (v_low**2 + v_high**2))
    terminal  = config.terminal_weight * (Q_C * e_cot[-1] ** 2 + Q_L * e_lag[-1] ** 2)
    return jnp.sum(stage + penalties) + terminal


def make_objective(
    bundle:     DeepONetBundle | None,
    track_data: TrackData,
    config:     MPCCPANOCConfig,
    horizon:    int,
    n_u:        int,
):
    def objective(u_flat, x0, p0):
        u_seq  = u_flat.reshape((horizon, n_u))
        x_pred = _predict_states_for_model(bundle, config.dynamics_model, x0, u_seq)
        return _mpcc_cost_from_prediction(x_pred, u_flat, p0, track_data, config, horizon, n_u)

    return objective


def make_scaled_value_and_grad(
    bundle: DeepONetBundle,
    track_data: TrackData,
    config: MPCCPANOCConfig,
    horizon: int,
    n_u: int,
    control_center: jnp.ndarray,
    control_half_range: jnp.ndarray,
):
    def value_and_grad(y_flat, x0, p0):
        y_seq = y_flat.reshape((horizon, n_u))
        u_seq = control_center + control_half_range * y_seq
        u_flat = u_seq.reshape(-1)
        x_pred = _predict_states_for_model(bundle, config.dynamics_model, x0, u_seq)

        cost_from_x_and_u = lambda xp, uf: _mpcc_cost_from_prediction(
            xp,
            uf,
            p0,
            track_data,
            config,
            horizon,
            n_u,
        )
        value, (grad_x, grad_u_direct) = jax.value_and_grad(
            cost_from_x_and_u,
            argnums=(0, 1),
        )(x_pred, u_flat)

        dx_du = _predict_states_u_jacobian_for_model(
            bundle, x0, u_seq, config.dynamics_model
        ).reshape(
            horizon * bundle.n_x,
            horizon * bundle.n_u,
        )
        grad_u_model = dx_du.T @ grad_x.reshape(-1)
        grad_u = grad_u_direct + grad_u_model
        grad_y = (grad_u.reshape((horizon, n_u)) * control_half_range).reshape(-1)
        return value, grad_y

    return value_and_grad


def make_deeponet_panoc_solver(
    bundle: DeepONetBundle | None,
    track_data: TrackData,
    config: MPCCPANOCConfig | None = None,
):
    config  = MPCCPANOCConfig() if config is None else config
    if config.dynamics_model not in (*DEEPONET_MODELS, "kinematic"):
        raise ValueError(f"Unknown dynamics_model={config.dynamics_model!r}")
    if config.dynamics_model in DEEPONET_MODELS and bundle is None:
        raise ValueError(f"A DeepONet bundle is required for dynamics_model={config.dynamics_model!r}")

    is_deeponet = config.dynamics_model in DEEPONET_MODELS
    horizon = int(bundle.horizon) if is_deeponet else H
    n_u = int(bundle.n_u) if is_deeponet else 2
    physical_lower = np.array([OMEGA_MIN, A_MIN], dtype=np.float32)
    physical_upper = np.array([OMEGA_MAX, A_MAX], dtype=np.float32)
    control_lower = physical_lower
    control_upper = physical_upper
    if np.any(control_upper <= control_lower):
        raise ValueError(f"Invalid PANOC control bounds: lower={control_lower}, upper={control_upper}")

    control_center = jnp.asarray(0.5 * (control_lower + control_upper), dtype=jnp.float32)
    control_half_range = jnp.asarray(0.5 * (control_upper - control_lower), dtype=jnp.float32)
    lower = np.full(horizon * n_u, -1.0, dtype=np.float32)
    upper = np.full(horizon * n_u, 1.0, dtype=np.float32)
    objective = make_objective(
        bundle=bundle,
        track_data=track_data,
        config=config,
        horizon=horizon,
        n_u=n_u,
    )

    def scaled_objective(y_flat, x0, p0):
        y_seq = y_flat.reshape((horizon, n_u))
        u_seq = control_center + control_half_range * y_seq
        return objective(u_seq.reshape(-1), x0, p0)

    value_and_grad_fn = None
    if bundle is not None and config.dynamics_model in DEEPONET_MODELS:
        value_and_grad_fn = make_scaled_value_and_grad(
            bundle,
            track_data,
            config,
            horizon,
            n_u,
            control_center,
            control_half_range,
        )

    solver = make_panoc_solver(
        scaled_objective,
        lower,
        upper,
        config.panoc,
        value_and_grad_fn=value_and_grad_fn,
    )

    def solve(x0: np.ndarray, p0: float, warm_u: np.ndarray | None = None):
        if warm_u is None:
            warm_y_arr = np.zeros((horizon, n_u), dtype=np.float32)
        else:
            warm_u_arr = np.asarray(warm_u, dtype=np.float32).reshape((horizon, n_u))
            warm_y_arr = np.asarray(
                (warm_u_arr - np.asarray(control_center)) / np.asarray(control_half_range),
                dtype=np.float32,
            )
            warm_y_arr = np.clip(warm_y_arr, -1.0, 1.0)
        x0_jax = jnp.asarray(x0, dtype=jnp.float32)
        result = solver(warm_y_arr.reshape(-1), x0_jax, jnp.asarray(float(p0), dtype=jnp.float32))
        y_sol = np.asarray(result["u"]).reshape((horizon, n_u))
        u_sol = np.asarray(control_center) + np.asarray(control_half_range) * y_sol
        x_pred = np.asarray(_predict_states_for_model(bundle, config.dynamics_model, x0_jax, jnp.asarray(u_sol)))
        v_pred = np.clip(x_pred[:, 3], V_MIN, V_MAX)
        s_sol = np.asarray(DT * v_pred, dtype=np.float32)
        p_sol = float(p0) + np.cumsum(s_sol)
        return {
            "X": x_pred,
            "U": u_sol,
            "P": p_sol,
            "S": s_sol,
            "objective": float(result["objective"]),
            "initial_residual_norm": float(result["initial_residual_norm"]),
            "residual_norm": float(result["residual_norm"]),
            "iterations": int(result["iterations"]),
            "converged": bool(result["converged"]),
            "control_lower": control_lower,
            "control_upper": control_upper,
        }

    return solve


def solve_deeponet_panoc(
    bundle: DeepONetBundle,
    track_data: TrackData,
    x0: np.ndarray,
    p0: float,
    warm_u: np.ndarray | None = None,
    config: MPCCPANOCConfig | None = None,
):
    config = MPCCPANOCConfig() if config is None else config
    solver = make_deeponet_panoc_solver(bundle, track_data, config)
    return solver(x0=x0, p0=p0, warm_u=warm_u)


def make_deeponet_no_root_panoc_solver(
    bundle: DeepONetBundle,
    track_data: TrackData,
    config: MPCCPANOCConfig | None = None,
):
    """Backward-compatible alias for the unified PANOC solver."""
    return make_deeponet_panoc_solver(bundle, track_data, config)
