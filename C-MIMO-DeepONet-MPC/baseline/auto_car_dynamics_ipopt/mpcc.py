"""
MPCC solvers using CasADi/IPOPT.

State and input follow AI_guideline_CTG:
    x = [X, Y, phi, v]
    u = [omega, a]

The path progress p and progress increment vartheta are separate optimization
variables. They are not part of x or u.
"""

from __future__ import annotations

import numpy as np
import casadi as ca
from pathlib import Path
import yaml

import track as _track
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MPCC_CONFIG = _REPO_ROOT / "model" / "auto_car" / "mpcc_parameter.yaml"

with open(_MPCC_CONFIG, "r") as file:
    param = yaml.safe_load(file)

H = int(param["H"])
DT = float(param["DT"])
DELTA_MAX = float(param["DELTA_MAX"])
n_grid = int(param["n_grid"])
ZETA_P = float(param["ZETA_P"])
OMEGA_MIN = float(param["OMEGA_MIN"])
OMEGA_MAX = float(param["OMEGA_MAX"])
A_MIN = float(param["A_MIN"])
A_MAX = float(param["A_MAX"])
V_MIN = float(param["V_MIN"])
V_MAX = float(param["V_MAX"])
Q_C = float(param["Q_C"])
Q_L = float(param["Q_L"])
Q_V = float(param["Q_V"])
Q_OMEGA = float(param["Q_OMEGA"])
Q_A = float(param["Q_A"])

transition_matrix = np.array([
    [0.8,   0.2],
    [0.2,   0.8],
])

_p_max  = _track.p_max
_PATH_INTERPOLANTS = None
_SOLVER_ACCEPT_TOL = 1e-3


def _eval_path(p_vals_mod: np.ndarray):
    """Evaluate Xcen, Ycen, Phi at arc-length values already wrapped by p_max."""
    Xc, Yc = _track.centerline_xy(p_vals_mod)
    Phi = np.array([_track.tangent_angle(np.array([s]))[0] for s in p_vals_mod])
    return Xc, Yc, Phi


def evaluate_path_geometry(p_vals: np.ndarray):
    """Evaluate centerline coordinates and tangent angle at progress values."""
    return _eval_path(np.asarray(p_vals, dtype=float) % _p_max)


def _get_path_interpolants():
    """Return CasADi interpolants for track geometry as functions of progress."""
    global _PATH_INTERPOLANTS
    if _PATH_INTERPOLANTS is not None:
        return _PATH_INTERPOLANTS

    grid   = np.linspace(0.0, _p_max, n_grid)
    Xc, Yc = _track.centerline_xy(grid)
    Phi    = np.unwrap(_track.tangent_angle(grid))

    # Match the reference MPCC pattern: remove the duplicate full-loop endpoint
    # and build a periodic extension so the interpolants stay smooth near wrap.
    grid = grid[:-1]
    Xc   = Xc[:-1]
    Yc   = Yc[:-1]
    Phi  = Phi[:-1]
    grid_ext = np.concatenate([grid - _p_max, grid, grid + _p_max])
    Xc_ext   = np.concatenate([Xc, Xc, Xc])
    Yc_ext   = np.concatenate([Yc, Yc, Yc])
    Phi_ext  = np.concatenate([Phi - 2.0 * np.pi, Phi, Phi + 2.0 * np.pi])

    def make_interpolant(name, values):
        values = np.asarray(values, dtype=float).reshape((-1,))
        try:
            return ca.interpolant(name, "bspline", [grid_ext], values)
        except RuntimeError:
            return ca.interpolant(name, "linear", [grid_ext], values)

    _PATH_INTERPOLANTS = {
        "X": make_interpolant("track_x_cen", Xc_ext),
        "Y": make_interpolant("track_y_cen", Yc_ext),
        "phi": make_interpolant("track_phi", Phi_ext),
    }
    return _PATH_INTERPOLANTS


def _local_progress_expr(p_expr, lap_origin):
    """Return smooth local progress relative to the current lap origin."""
    return p_expr - lap_origin


def evaluate_stage_costs(
    X_sol: np.ndarray,
    U_sol: np.ndarray,
    vartheta_sol: np.ndarray,
    path_geometry: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Evaluate MPCC stage costs for solved 4D state and 2D input trajectories."""
    Xcen, Ycen, Phi = path_geometry
    X_arr = np.asarray(X_sol, dtype=float)
    U_arr = np.asarray(U_sol, dtype=float)
    vartheta_arr = np.asarray(vartheta_sol, dtype=float).reshape((-1,))
    costs = []
    for i in range(len(U_arr)):
        dx = X_arr[i, 0] - Xcen[i]
        dy = X_arr[i, 1] - Ycen[i]
        sin_phi = np.sin(Phi[i])
        cos_phi = np.cos(Phi[i])
        e_cot = sin_phi * dx - cos_phi * dy
        e_lag = -cos_phi * dx - sin_phi * dy
        omega_i, a_i = U_arr[i]
        costs.append(
            Q_C * e_cot**2
            + Q_L * e_lag**2
            + Q_OMEGA * omega_i**2
            + Q_A * a_i**2
            - Q_V * vartheta_arr[i]
        )
    return np.asarray(costs, dtype=float)


def _debug_solution_is_acceptable(
    x0: np.ndarray,
    p0: float,
    v_max: float,
    X_sol: np.ndarray,
    U_sol: np.ndarray,
    P_sol: np.ndarray,
    S_sol: np.ndarray,
) -> bool:
    """Accept an IPOPT debug solution only when constraints are numerically met."""
    X_arr = np.asarray(X_sol, dtype=float)
    U_arr = np.asarray(U_sol, dtype=float)
    P_arr = np.asarray(P_sol, dtype=float).reshape((-1,))
    S_arr = np.asarray(S_sol, dtype=float).reshape((-1,))
    x_prev = np.asarray(x0, dtype=float)
    p_prev = None

    if len(X_arr) == 0 or len(U_arr) != len(X_arr) or len(P_arr) != len(X_arr) or len(S_arr) != len(X_arr):
        return False
    if not (np.all(np.isfinite(X_arr)) and np.all(np.isfinite(U_arr)) and np.all(np.isfinite(P_arr)) and np.all(np.isfinite(S_arr))):
        return False

    Xcen, Ycen, _ = evaluate_path_geometry(P_arr)
    p_prev = float(p0)
    for i in range(len(X_arr)):
        omega_i, a_i = U_arr[i]
        dyn_next = np.array([
            x_prev[0] + DT * x_prev[3] * np.cos(x_prev[2]),
            x_prev[1] + DT * x_prev[3] * np.sin(x_prev[2]),
            x_prev[2] + DT * omega_i,
            x_prev[3] + DT * ZETA_P * a_i,
        ])
        if np.max(np.abs(X_arr[i] - dyn_next)) > _SOLVER_ACCEPT_TOL:
            return False

        if not (OMEGA_MIN - _SOLVER_ACCEPT_TOL <= omega_i <= OMEGA_MAX + _SOLVER_ACCEPT_TOL):
            return False
        if not (A_MIN - _SOLVER_ACCEPT_TOL <= a_i <= A_MAX + _SOLVER_ACCEPT_TOL):
            return False
        if not (V_MIN - _SOLVER_ACCEPT_TOL <= X_arr[i, 3] <= v_max + _SOLVER_ACCEPT_TOL):
            return False
        if not (0.0 - _SOLVER_ACCEPT_TOL <= S_arr[i] <= DT * X_arr[i, 3] + _SOLVER_ACCEPT_TOL):
            return False
        if abs(P_arr[i] - (p_prev + S_arr[i])) > _SOLVER_ACCEPT_TOL:
            return False

        dist = np.hypot(X_arr[i, 0] - Xcen[i], X_arr[i, 1] - Ycen[i])
        if dist > DELTA_MAX + _SOLVER_ACCEPT_TOL:
            return False

        x_prev = X_arr[i]
        p_prev = P_arr[i]
    return True


def evaluate_total_cost(
    p0: float,
    X_sol: np.ndarray,
    U_sol: np.ndarray,
    vartheta_sol: np.ndarray,
) -> float:
    """Return the OP1/OP2 stage-cost sum for a solved trajectory."""
    geometry = path_geometry_for_op1(p0, vartheta_sol)
    return float(np.sum(evaluate_stage_costs(X_sol, U_sol, vartheta_sol, geometry)))


def infer_progress_from_state(x0: np.ndarray) -> float:
    """Derive p0 from 4D state by solving OP0."""
    x_arr = np.asarray(x0, dtype=float)
    return float(_track.find_progress(float(x_arr[0]), float(x_arr[1]), 0.0, n_search=1, n_grid=n_grid))



def _build_stage_solver(horizon: int):
    opti = ca.Opti()
    opti.solver(
        "ipopt",
        {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 1000,
            "ipopt.tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
        },
    )

    X_var = opti.variable(horizon, 4)  # [X, Y, phi, v]
    U_var = opti.variable(horizon, 2)  # [omega, a]
    P_var = opti.variable(horizon)     # p_{1|t}, ..., p_{H|t}
    S_var = opti.variable(horizon)     # vartheta_{0|t}, ..., vartheta_{H-1|t}

    x0_p     = opti.parameter(4)
    p0_p     = opti.parameter(1)
    lap_origin_p = opti.parameter(1)
    theta_p  = opti.parameter(1)
    v_max_p  = opti.parameter(1)
    path_lut = _get_path_interpolants()

    q_v = theta_p[0]
    stage_cost = 0.0
    x_prev = x0_p
    p_prev = p0_p[0]

    for i in range(horizon):
        Xi, Yi, phi_i, vi = (X_var[i, j] for j in range(4))
        omega_i, a_i = (U_var[i, j] for j in range(2))
        p_i = P_var[i]
        vartheta_i = S_var[i]
        X_p, Y_p, phi_p, v_p = (x_prev[j] for j in range(4))

        opti.subject_to(Xi == X_p + DT * v_p * ca.cos(phi_p))
        opti.subject_to(Yi == Y_p + DT * v_p * ca.sin(phi_p))
        opti.subject_to(phi_i == phi_p + DT * omega_i)
        opti.subject_to(vi == v_p + DT * ZETA_P * a_i)
        opti.subject_to(p_i == p_prev + vartheta_i)

        p_ref   = _local_progress_expr(p_i, lap_origin_p[0])
        Xcen_i  = path_lut["X"](p_ref)
        Ycen_i  = path_lut["Y"](p_ref)
        phi_ref = path_lut["phi"](p_ref)
        sin_phi = ca.sin(phi_ref)
        cos_phi = ca.cos(phi_ref)
        dx      = Xi - Xcen_i
        dy      = Yi - Ycen_i
        e_cot = sin_phi * dx - cos_phi * dy
        e_lag = -cos_phi * dx - sin_phi * dy
        stage_cost += (
            Q_C * e_cot**2
            + Q_L * e_lag**2
            + Q_OMEGA * omega_i**2
            + Q_A * a_i**2
            - q_v * vartheta_i
        )

        opti.subject_to(opti.bounded(OMEGA_MIN, omega_i, OMEGA_MAX))
        opti.subject_to(opti.bounded(A_MIN, a_i, A_MAX))
        opti.subject_to(opti.bounded(0, vartheta_i, DT * vi))
        opti.subject_to(p_i >= 0)
        opti.subject_to(opti.bounded(V_MIN, vi, v_max_p[0]))
        opti.subject_to(dx**2 + dy**2 <= DELTA_MAX**2)

        x_prev = X_var[i, :]
        p_prev = p_i

    # total_cost = stage_cost + _terminal_ctg_expr(X_var[horizon - 1, :], q_v, model_bundle)
    total_cost = stage_cost
    opti.minimize(total_cost)

    handles = dict(
        X_var=X_var,
        U_var=U_var,
        P_var=P_var,
        S_var=S_var,
        x0_p=x0_p,
        p0_p=p0_p,
        lap_origin_p=lap_origin_p,
        theta_p=theta_p,
        v_max_p=v_max_p,
        stage_cost=stage_cost,
        total_cost=total_cost,
    )
    return opti, handles


_SOLVER_CACHE = {}


def _get_stage_solver(horizon: int):
    if horizon not in _SOLVER_CACHE:
        _SOLVER_CACHE[horizon] = _build_stage_solver(horizon)
    return _SOLVER_CACHE[horizon]


_opti, _h = _get_stage_solver(H)


def _make_warm_start(x0: np.ndarray, p0: float, horizon: int = H, v_max: float = V_MAX):
    vartheta0 = DT * v_max / 2
    X_g = np.zeros((horizon, 4))
    U_g = np.zeros((horizon, 2))
    P_g = np.zeros(horizon)
    S_g = np.full(horizon, vartheta0)
    x = np.asarray(x0, dtype=float).copy()
    p = float(p0)

    for i in range(horizon):
        U_g[i] = np.array([0.0, 0.0])
        X_p, Y_p, phi_p, v_p = x
        x = np.array([
            X_p + DT * v_p * np.cos(phi_p),
            Y_p + DT * v_p * np.sin(phi_p),
            phi_p,
            np.clip(v_p, V_MIN, v_max),
        ])
        p += vartheta0
        X_g[i] = x
        P_g[i] = p
    return X_g, U_g, P_g, S_g


def path_geometry_for_op1(
    p0: float,
    warm_vartheta: np.ndarray | None = None,
    horizon: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return numeric path geometry for diagnostics after a trajectory is known."""
    if warm_vartheta is None:
        h = H if horizon is None else horizon
        warm_vartheta = np.full(h, DT * V_MAX / 2)
    p_steps = float(p0) + np.cumsum(np.asarray(warm_vartheta, dtype=float))
    return _eval_path(p_steps % _p_max)


def _set_and_solve(
    opti,
    h,
    x0: np.ndarray,
    p0: float,
    v_max: float,
    warm_X: np.ndarray,
    warm_U: np.ndarray,
    warm_P: np.ndarray,
    warm_vartheta: np.ndarray,
):
    horizon = len(warm_vartheta)
    lap_origin = np.floor(float(p0) / _p_max) * _p_max

    opti.set_value(h["x0_p"], np.asarray(x0, dtype=float))
    opti.set_value(h["p0_p"], [float(p0)])
    opti.set_value(h["lap_origin_p"], [lap_origin])
    opti.set_value(h["theta_p"], [Q_V])
    opti.set_value(h["v_max_p"], [float(v_max)])
    opti.set_initial(h["X_var"], warm_X)
    opti.set_initial(h["U_var"], warm_U)
    opti.set_initial(h["P_var"], warm_P)
    opti.set_initial(h["S_var"], warm_vartheta)

    try:
        sol = opti.solve()
        return (
            np.array(sol.value(h["X_var"])),
            np.array(sol.value(h["U_var"])),
            np.array(sol.value(h["P_var"])).reshape((-1,)),
            np.array(sol.value(h["S_var"])).reshape((-1,)),
            float(sol.value(h["stage_cost"])),
            True,
        )
    except Exception:
        try:
            X_dbg = np.array(opti.debug.value(h["X_var"]))
            U_dbg = np.array(opti.debug.value(h["U_var"]))
            P_dbg = np.array(opti.debug.value(h["P_var"])).reshape((-1,))
            S_dbg = np.array(opti.debug.value(h["S_var"])).reshape((-1,))
            value_dbg = float(opti.debug.value(h["stage_cost"]))
            acceptable = _debug_solution_is_acceptable(
                x0=x0,
                p0=p0,
                v_max=v_max,
                X_sol=X_dbg,
                U_sol=U_dbg,
                P_sol=P_dbg,
                S_sol=S_dbg,
            )
            return (
                X_dbg,
                U_dbg,
                P_dbg,
                S_dbg,
                value_dbg,
                acceptable,
            )
        except Exception:
            return warm_X, warm_U, warm_P, warm_vartheta, float("nan"), False


def _coerce_warm_start(x0, p0, warm_X, warm_U, warm_P, warm_vartheta, horizon, v_max: float = V_MAX):
    if warm_X is None or warm_U is None or warm_P is None or warm_vartheta is None:
        return _make_warm_start(x0, p0, horizon=horizon, v_max=v_max)
    warm_X_arr = np.asarray(warm_X, dtype=float)[:horizon].copy()
    warm_X_arr[:, 3] = np.clip(warm_X_arr[:, 3], V_MIN, v_max)
    warm_vartheta_arr = np.asarray(warm_vartheta, dtype=float).reshape((-1,))[:horizon].copy()
    warm_vartheta_arr = np.clip(warm_vartheta_arr, 0.0, DT * warm_X_arr[:, 3])
    warm_P_arr = np.asarray(warm_P, dtype=float).reshape((-1,))[:horizon].copy()
    if len(warm_P_arr) > 0:
        p0_float = float(p0)
        max_forward_guess = horizon * DT * v_max + _SOLVER_ACCEPT_TOL
        while warm_P_arr[0] - p0_float > max_forward_guess:
            warm_P_arr -= _p_max
        if warm_P_arr[0] < p0_float:
            warm_P_arr += p0_float - warm_P_arr[0]
    return (
        warm_X_arr,
        np.asarray(warm_U, dtype=float)[:horizon],
        warm_P_arr,
        warm_vartheta_arr,
    )


def solve_op1(x0: np.ndarray,
              p0: float,
              curr_state: float = 1,
              warm_X: np.ndarray | None = None,
              warm_U: np.ndarray | None = None,
              warm_P: np.ndarray | None = None,
              warm_vartheta: np.ndarray | None = None,
              v_max: float = V_MAX,
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Solve OP1 with 4D state, 2D input, and separate progress variables."""
    warm_X, warm_U, warm_P, warm_vartheta = _coerce_warm_start(
        x0, p0, warm_X, warm_U, warm_P, warm_vartheta, H, v_max=v_max
    )

    X_sol, U_sol, P_sol, S_sol, _, ok = _set_and_solve(
        _opti, _h, x0, p0, curr_state, v_max, warm_X, warm_U, warm_P, warm_vartheta
    )
    return X_sol, U_sol, P_sol, S_sol, ok


def solve_op1_horizon(x0: np.ndarray,
                      horizon: int,
                      warm_X: np.ndarray | None = None,
                      warm_U: np.ndarray | None = None,
                      warm_P: np.ndarray | None = None,
                      warm_vartheta: np.ndarray | None = None,
                      v_max: float = V_MAX,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Solve a stage-only OP1 problem for an arbitrary horizon."""
    p0 = infer_progress_from_state(x0)
    warm_X, warm_U, warm_P, warm_vartheta = _coerce_warm_start(
        x0, p0, warm_X, warm_U, warm_P, warm_vartheta, horizon, v_max=v_max
    )

    opti, h = _get_stage_solver(horizon)
    return _set_and_solve(opti, h, x0, p0, v_max, warm_X, warm_U, warm_P, warm_vartheta)
