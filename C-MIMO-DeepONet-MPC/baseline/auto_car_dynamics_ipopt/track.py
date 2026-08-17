"""
Reference path for the Oschersleben racetrack.

The centerline is parameterised by arc-length s ∈ [0, p_max].
All public functions accept scalar or array s values.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import os
import importlib.util
from pathlib import Path
import yaml

REPO_ROOT   = Path(__file__).resolve().parents[2]
MPCC_CONFIG = REPO_ROOT / "model" / "auto_car" / "mpcc_parameter.yaml"
with open(MPCC_CONFIG, 'r') as file:
    param = yaml.safe_load(file)


TRACK_CSV_PATH = REPO_ROOT / param["TRACK_CSV"]


def _load_and_build(csv_path: str):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    x_wp    = data[:, 0]
    y_wp    = data[:, 1]
    w_right = data[:, 2]   # track half-width right [m]
    w_left  = data[:, 3]   # track half-width left  [m]

    # Close the loop: append the first waypoint at the end
    x_wp    = np.append(x_wp,  x_wp[0])
    y_wp    = np.append(y_wp,  y_wp[0])
    w_right = np.append(w_right, w_right[0])
    w_left  = np.append(w_left,  w_left[0])

    # Arc-length parameterisation
    dx = np.diff(x_wp)
    dy = np.diff(y_wp)
    ds = np.hypot(dx, dy)
    s_wp  = np.concatenate([[0.0], np.cumsum(ds)])
    p_max = float(s_wp[-1])

    # Periodic cubic spline (closed track)
    cs_x = CubicSpline(s_wp, x_wp, bc_type="periodic")
    cs_y = CubicSpline(s_wp, y_wp, bc_type="periodic")

    # Interpolated track-width (constant here but kept general)
    cs_wr = CubicSpline(s_wp, w_right, bc_type="periodic")
    cs_wl = CubicSpline(s_wp, w_left,  bc_type="periodic")

    return cs_x, cs_y, cs_wr, cs_wl, p_max


# ── Module-level singleton (loaded once) ────────────────────────────────────
_CSV = os.fspath(TRACK_CSV_PATH)
_cs_x, _cs_y, _cs_wr, _cs_wl, p_max = _load_and_build(_CSV)


def centerline_xy(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_cen, Y_cen) at arc-length s (mod p_max applied externally)."""
    return _cs_x(s), _cs_y(s)


def tangent_angle(s: np.ndarray) -> np.ndarray:
    """Φ(s) = atan2(dY/ds, dX/ds) — heading of the path tangent."""
    dxds = _cs_x(s, 1)
    dyds = _cs_y(s, 1)
    return np.arctan2(dyds, dxds)


def track_widths(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (w_right, w_left) half-widths at arc-length s."""
    return _cs_wr(s), _cs_wl(s)


def find_progress(X: float,
                  Y: float,
                  p_guess: float,
                  n_search: int = 5,
                  n_grid: int = 101) -> float:
    """
    OP0: find p̄ ∈ [0, p_max] that minimises distance to (X, Y).

    Uses a coarse grid search near p_guess then refines with scipy.
    """
    from scipy.optimize import minimize_scalar

    # Search window: ±(p_max/n_search) around guess. The distance along a
    # long track segment is not unimodal, so sample first and refine locally.
    width = p_max / n_search

    def dist2(s):
        xc, yc = centerline_xy(np.array([s % p_max]))
        return (X - xc[0]) ** 2 + (Y - yc[0]) ** 2

    lo = p_guess - width
    hi = p_guess + width
    grid = np.linspace(lo, hi, n_grid)
    vals = np.array([dist2(s) for s in grid])
    best = int(np.argmin(vals))
    step = grid[1] - grid[0] if n_grid > 1 else width
    res = minimize_scalar(
        dist2,
        bounds=(grid[best] - step, grid[best] + step),
        method="bounded",
    )

    return float(res.x % p_max)


def contour_lag_errors(X: float, Y: float, s: float) -> tuple[float, float]:
    """
    e_cot and e_lag for position (X, Y) at path progress s.
    """
    xc, yc = centerline_xy(np.array([s]))
    phi = tangent_angle(np.array([s]))
    dx = X - xc[0]
    dy = Y - yc[0]
    sin_phi = np.sin(phi[0])
    cos_phi = np.cos(phi[0])
    e_cot =  sin_phi * dx - cos_phi * dy
    e_lag = -cos_phi * dx - sin_phi * dy
    return float(e_cot), float(e_lag)


def border_lines(n_pts: int = 500):
    """
    Return left and right borderline XY arrays for plotting.
    Each is shape (n_pts, 2).
    """
    s_vals = np.linspace(0.0, p_max, n_pts)
    xc, yc = centerline_xy(s_vals)
    phi    = tangent_angle(s_vals)
    wr, wl = track_widths(s_vals)

    # Normal direction (perpendicular to tangent, pointing left)
    nx = -np.sin(phi)
    ny =  np.cos(phi)

    right = np.stack([xc - wr * nx, yc - wr * ny], axis=1)
    left  = np.stack([xc + wl * nx, yc + wl * ny], axis=1)
    center = np.stack([xc, yc], axis=1)
    return center, left, right
