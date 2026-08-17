"""CasADi/IPOPT MPCC solver for C-MIMO DeepONet predictions."""

from pathlib import Path
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FILE_DIR = REPO_ROOT / "baseline" / "auto_car_dynamics_ipopt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .mpcc import MPCCPANOCConfig
import baseline.auto_car_dynamics_ipopt.track

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
n_grid = int(param["n_grid"])


def _control_bounds(bundle, config: MPCCPANOCConfig):
    physical_lower = np.array([OMEGA_MIN, A_MIN], dtype=np.float32)
    physical_upper = np.array([OMEGA_MAX, A_MAX], dtype=np.float32)
    control_lower = physical_lower
    control_upper = physical_upper
    if np.any(control_upper <= control_lower):
        raise ValueError(f"Invalid control bounds: lower={control_lower}, upper={control_upper}")
    return control_lower, control_upper


_CASADI_TRACK_INTERPOLANTS = None


def _get_casadi_track_interpolants():
    global _CASADI_TRACK_INTERPOLANTS
    if _CASADI_TRACK_INTERPOLANTS is not None:
        return _CASADI_TRACK_INTERPOLANTS

    import casadi as ca

    grid = np.linspace(0.0, float(track.p_max), n_grid)
    x_cen, y_cen = track.centerline_xy(grid)
    phi_cen = np.unwrap(track.tangent_angle(grid))

    grid = grid[:-1]
    x_cen = x_cen[:-1]
    y_cen = y_cen[:-1]
    phi_cen = phi_cen[:-1]
    grid_ext = np.concatenate([grid - track.p_max, grid, grid + track.p_max])
    x_ext = np.concatenate([x_cen, x_cen, x_cen])
    y_ext = np.concatenate([y_cen, y_cen, y_cen])
    phi_ext = np.concatenate([phi_cen - 2.0 * np.pi, phi_cen, phi_cen + 2.0 * np.pi])

    def make_interpolant(name, values):
        values = np.asarray(values, dtype=float).reshape((-1,))
        try:
            return ca.interpolant(name, "bspline", [grid_ext], values)
        except RuntimeError:
            return ca.interpolant(name, "linear", [grid_ext], values)

    _CASADI_TRACK_INTERPOLANTS = {
        "X": make_interpolant("deeponet_ipopt_track_x", x_ext),
        "Y": make_interpolant("deeponet_ipopt_track_y", y_ext),
        "phi": make_interpolant("deeponet_ipopt_track_phi", phi_ext),
    }
    return _CASADI_TRACK_INTERPOLANTS


def _casadi_leaky_relu(ca, x, negative_slope: float = 0.01):
    return ca.fmax(x, negative_slope * x)


def _casadi_dense(ca, layer, x):
    w = ca.DM(np.asarray(layer["w"], dtype=float))
    b = ca.DM(np.asarray(layer["b"], dtype=float).reshape((1, -1)))
    return ca.mtimes(x, w) + b


def _casadi_mlp3h(ca, layers, x):
    x = _casadi_leaky_relu(ca, _casadi_dense(ca, layers[0], x))
    x = _casadi_leaky_relu(ca, _casadi_dense(ca, layers[1], x))
    return _casadi_dense(ca, layers[2], x)


def _hard_sigmoid_np(x):
    return np.clip(x / 6.0 + 0.5, 0.0, 1.0)


def _casadi_root_initial_features(ca, x0_row, n_x: int, root_input_dim: int):
    if root_input_dim == n_x:
        return x0_row
    if root_input_dim != n_x + 1:
        raise ValueError(
            f"Unsupported root input dimension {root_input_dim} for state dimension {n_x}"
        )

    return ca.horzcat(
        x0_row[:, :2],
        ca.sin(x0_row[:, 2]),
        ca.cos(x0_row[:, 2]),
        x0_row[:, 3:],
    )


def _casadi_predict_deeponet_states(ca, bundle, x0, u_seq):
    x0_row = ca.reshape(x0, 1, bundle.n_x)
    u_flat = ca.reshape(u_seq.T, 1, bundle.horizon * bundle.n_u)

    branch_outputs = []
    for branch_params, (start_idx, end_idx) in zip(bundle.params["branches"], bundle.branch_ranges):
        branch_input = u_flat[:, start_idx:end_idx]
        branch_outputs.append(_casadi_mlp3h(ca, branch_params, branch_input))
    root_input_dim = bundle.root_input_dim
    if root_input_dim is None:
        root_input_dim = int(np.asarray(bundle.params["root"][0]["w"]).shape[0])
    root_input = _casadi_root_initial_features(ca, x0_row, bundle.n_x, root_input_dim)
    root_flat = _casadi_mlp3h(ca, bundle.params["root"], root_input)

    outputs = []
    num_pred = len(bundle.branch_ranges)
    branch_idx = np.arange(1, num_pred + 1, dtype=float)
    for i in range(num_pred):
        t_i = float(i + 1)
        gate = _hard_sigmoid_np(10.0 * (branch_idx - t_i + 0.5)) - _hard_sigmoid_np(10.0 * (branch_idx - t_i - 0.5))
        trunk_flat = _casadi_mlp3h(ca, bundle.params["trunk"], ca.DM([[t_i]]))
        row = []
        for state_idx in range(bundle.n_x):
            y_i = x0_row[0, state_idx]
            for latent_idx in range(bundle.p):
                flat_idx = latent_idx * bundle.n_x + state_idx
                branch_star = 0.0
                for branch_out, gate_j in zip(branch_outputs, gate):
                    branch_star += float(gate_j) * branch_out[0, flat_idx]
                y_i += branch_star * trunk_flat[0, flat_idx] * root_flat[0, flat_idx]
            row.append(y_i)
        outputs.append(ca.horzcat(*row))
    return ca.vertcat(*outputs)


def _make_ipopt_warm_start(
    x0: np.ndarray,
    p0: float,
    horizon: int,
    control_lower: np.ndarray,
    control_upper: np.ndarray,
):
    u_guess = np.zeros((horizon, 2), dtype=float)
    u_guess = np.clip(u_guess, control_lower, control_upper)
    x_guess = np.zeros((horizon, 4), dtype=float)
    p_guess = np.zeros(horizon, dtype=float)
    s_guess = np.zeros(horizon, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    p = float(p0)

    for i in range(horizon):
        omega_i, accel_i = u_guess[i]
        x = np.array(
            [
                x[0] + DT * x[3] * np.cos(x[2]),
                x[1] + DT * x[3] * np.sin(x[2]),
                x[2] + DT * omega_i,
                np.clip(x[3] + DT * ZETA_P * accel_i, V_MIN, V_MAX),
            ],
            dtype=float,
        )
        s_i = np.clip(DT * x[3], 0.0, DT * V_MAX)
        p += s_i
        x_guess[i] = x
        p_guess[i] = p
        s_guess[i] = s_i

    return x_guess, u_guess, p_guess, s_guess


def _coerce_ipopt_warm_start(
    x0: np.ndarray,
    p0: float,
    warm_start,
    horizon: int,
    control_lower: np.ndarray,
    control_upper: np.ndarray,
):
    if isinstance(warm_start, dict) and all(k in warm_start for k in ("X", "U", "P", "S")):
        x_guess = np.asarray(warm_start["X"], dtype=float).reshape((horizon, 4)).copy()
        u_guess = np.asarray(warm_start["U"], dtype=float).reshape((horizon, 2)).copy()
        p_guess = np.asarray(warm_start["P"], dtype=float).reshape((horizon,)).copy()
        s_guess = np.asarray(warm_start["S"], dtype=float).reshape((horizon,)).copy()
    else:
        x_guess, u_guess, p_guess, s_guess = _make_ipopt_warm_start(
            x0=x0,
            p0=p0,
            horizon=horizon,
            control_lower=control_lower,
            control_upper=control_upper,
        )
        if warm_start is not None:
            u_guess = np.asarray(warm_start, dtype=float).reshape((horizon, 2)).copy()

    x_guess[:, 3] = np.clip(x_guess[:, 3], V_MIN, V_MAX)
    u_guess = np.clip(u_guess, control_lower, control_upper)
    s_guess = np.clip(s_guess, 0.0, DT * x_guess[:, 3])

    p0_float = float(p0)
    if len(p_guess) > 0:
        max_forward_guess = horizon * DT * V_MAX + 1e-3
        while p_guess[0] - p0_float > max_forward_guess:
            p_guess -= track.p_max
        if p_guess[0] < p0_float:
            p_guess += p0_float - p_guess[0]

    return x_guess, u_guess, p_guess, s_guess


def make_deeponet_ipopt_solver(
    bundle,
    config: MPCCPANOCConfig,
    max_iter: int = 1000,
    tol: float = 1e-4,
):
    try:
        import casadi as ca
    except ImportError as exc:
        raise RuntimeError("CasADi is required for --solver ipopt. Install casadi and rerun.") from exc

    if config.dynamics_model not in ("deeponet", "kinematic"):
        raise ValueError(f"Unknown dynamics_model={config.dynamics_model!r}")
    if config.dynamics_model == "deeponet" and bundle is None:
        raise ValueError("A DeepONet bundle is required when dynamics_model='deeponet'")

    horizon = int(bundle.horizon) if bundle is not None and config.dynamics_model == "deeponet" else H
    n_u = int(bundle.n_u) if bundle is not None and config.dynamics_model == "deeponet" else 2
    control_lower, control_upper = _control_bounds(bundle, config)

    if n_u != 2:
        raise ValueError(f"IPOPT MPCC expects controls [omega, a], got n_u={n_u}")

    opti = ca.Opti()
    opti.solver(
        "ipopt",
        {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": int(max_iter),
            "ipopt.tol": float(tol),
            "ipopt.warm_start_init_point": "yes",
        },
    )

    X_var = opti.variable(horizon, 4)
    U_var = opti.variable(horizon, 2)
    P_var = opti.variable(horizon)
    S_var = opti.variable(horizon)

    x0_p = opti.parameter(4)
    p0_p = opti.parameter(1)
    lap_origin_p = opti.parameter(1)
    theta_p = opti.parameter(1)
    v_max_p = opti.parameter(1)

    if config.dynamics_model == "deeponet":
        deeponet_pred = _casadi_predict_deeponet_states(ca, bundle, x0_p, U_var)
    else:
        deeponet_pred = None

    path_lut = _get_casadi_track_interpolants()
    stage_cost = 0.0
    x_prev = x0_p
    p_prev = p0_p[0]
    q_v = theta_p[0]

    for i in range(horizon):
        Xi, Yi, phi_i, vi = (X_var[i, j] for j in range(4))
        omega_i, accel_i = (U_var[i, j] for j in range(2))
        p_i = P_var[i]
        vartheta_i = S_var[i]

        if config.dynamics_model == "deeponet":
            for j in range(4):
                opti.subject_to(X_var[i, j] == deeponet_pred[i, j])
        else:
            X_p, Y_p, phi_p, v_p = (x_prev[j] for j in range(4))
            opti.subject_to(Xi == X_p + DT * v_p * ca.cos(phi_p))
            opti.subject_to(Yi == Y_p + DT * v_p * ca.sin(phi_p))
            opti.subject_to(phi_i == phi_p + DT * omega_i)
            opti.subject_to(vi == v_p + DT * ZETA_P * accel_i)

        opti.subject_to(p_i == p_prev + vartheta_i)

        p_ref = p_i - lap_origin_p[0]
        x_ref = path_lut["X"](p_ref)
        y_ref = path_lut["Y"](p_ref)
        phi_ref = path_lut["phi"](p_ref)
        dx = Xi - x_ref
        dy = Yi - y_ref
        sin_phi = ca.sin(phi_ref)
        cos_phi = ca.cos(phi_ref)
        e_cot = sin_phi * dx - cos_phi * dy
        e_lag = -cos_phi * dx - sin_phi * dy

        stage_cost += (
            Q_C * e_cot**2
            + Q_L * e_lag**2
            + Q_OMEGA * omega_i**2
            + Q_A * accel_i**2
            - q_v * vartheta_i
        )

        opti.subject_to(opti.bounded(float(control_lower[0]), omega_i, float(control_upper[0])))
        opti.subject_to(opti.bounded(float(control_lower[1]), accel_i, float(control_upper[1])))
        opti.subject_to(opti.bounded(0, vartheta_i, DT * vi))
        opti.subject_to(p_i >= 0)
        opti.subject_to(opti.bounded(V_MIN, vi, v_max_p[0]))
        opti.subject_to(dx**2 + dy**2 <= DELTA_MAX**2)

        x_prev = X_var[i, :]
        p_prev = p_i

    opti.minimize(stage_cost)

    def solve(x0: np.ndarray, p0: float, warm_u=None):
        warm_X, warm_U, warm_P, warm_S = _coerce_ipopt_warm_start(
            x0=x0,
            p0=p0,
            warm_start=warm_u,
            horizon=horizon,
            control_lower=control_lower,
            control_upper=control_upper,
        )

        opti.set_value(x0_p, np.asarray(x0, dtype=float))
        opti.set_value(p0_p, [float(p0)])
        opti.set_value(lap_origin_p, [np.floor(float(p0) / track.p_max) * track.p_max])
        opti.set_value(theta_p, [Q_V])
        opti.set_value(v_max_p, [V_MAX])
        opti.set_initial(X_var, warm_X)
        opti.set_initial(U_var, warm_U)
        opti.set_initial(P_var, warm_P)
        opti.set_initial(S_var, warm_S)

        converged = True
        try:
            sol = opti.solve()
            x_sol = np.asarray(sol.value(X_var), dtype=np.float32)
            u_sol = np.asarray(sol.value(U_var), dtype=np.float32)
            p_sol = np.asarray(sol.value(P_var), dtype=np.float32).reshape((-1,))
            s_sol = np.asarray(sol.value(S_var), dtype=np.float32).reshape((-1,))
            objective_value = float(sol.value(stage_cost))
        except Exception:
            converged = False
            try:
                x_sol = np.asarray(opti.debug.value(X_var), dtype=np.float32)
                u_sol = np.asarray(opti.debug.value(U_var), dtype=np.float32)
                p_sol = np.asarray(opti.debug.value(P_var), dtype=np.float32).reshape((-1,))
                s_sol = np.asarray(opti.debug.value(S_var), dtype=np.float32).reshape((-1,))
                objective_value = float(opti.debug.value(stage_cost))
            except Exception:
                x_sol = warm_X.astype(np.float32)
                u_sol = warm_U.astype(np.float32)
                p_sol = warm_P.astype(np.float32)
                s_sol = warm_S.astype(np.float32)
                objective_value = float("nan")

        try:
            stats = opti.stats()
            iterations = int(stats.get("iter_count", 0))
        except Exception:
            iterations = 0

        return {
            "X": x_sol,
            "U": u_sol,
            "P": p_sol,
            "S": s_sol,
            "objective": objective_value,
            "initial_residual_norm": np.nan,
            "residual_norm": 0.0 if converged else np.inf,
            "iterations": iterations,
            "converged": converged,
            "control_lower": control_lower,
            "control_upper": control_upper,
        }

    return solve
