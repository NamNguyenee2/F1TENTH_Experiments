import numpy as np
from scipy.interpolate import CubicSpline

HOME_POSE = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
JOINT_LIMITS = np.array([
    [-6.2832, 6.2832],
    [-6.2832, 6.2832],
    [-3.1416, 3.1416],
    [-6.2832, 6.2832],
    [-6.2832, 6.2832],
    [-6.2832, 6.2832],
])


def generate(rng: np.random.Generator, n_waypoints: int, duration: float,
             dt: float, joint_range_fraction: float = 0.6) -> dict:
    """Random joint-space trajectory via clamped cubic spline (starts/ends at home)."""
    t_wp = np.linspace(0.0, duration, n_waypoints)
    q_wp = np.zeros((n_waypoints, 6))
    q_wp[0] = HOME_POSE
    q_wp[-1] = HOME_POSE
    for j in range(6):
        lo = JOINT_LIMITS[j, 0] * joint_range_fraction
        hi = JOINT_LIMITS[j, 1] * joint_range_fraction
        q_wp[1:-1, j] = rng.uniform(lo, hi, n_waypoints - 2)
    cs = CubicSpline(t_wp, q_wp, bc_type="clamped")
    t = np.arange(0.0, duration, dt)
    return {"t": t, "q": cs(t), "qdot": cs(t, 1), "qdotdot": cs(t, 2)}
