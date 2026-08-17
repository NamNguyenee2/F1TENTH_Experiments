import numpy as np
from scipy.interpolate import CubicSpline

HOME_POSE = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

# Joint limits (safe 50% fraction)
JOINT_LIMITS = np.array([
    [-6.2832, 6.2832],
    [-6.2832, 6.2832],
    [-3.1416, 3.1416],
    [-6.2832, 6.2832],
    [-6.2832, 6.2832],
    [-6.2832, 6.2832],
])


class TrajectoryGenerator:
    """Generates joint-space trajectories for the UR3."""

    def single_joint_sine(
        self,
        joint_idx: int,
        amplitude: float,
        frequency: float,
        duration: float,
        dt: float,
    ) -> dict:
        """Sinusoidal trajectory for a single joint; others stay at home."""
        t = np.arange(0.0, duration, dt)
        n = len(t)
        q = np.tile(HOME_POSE, (n, 1))
        qdot = np.zeros((n, 6))
        qdotdot = np.zeros((n, 6))

        omega = 2.0 * np.pi * frequency
        q[:, joint_idx] = HOME_POSE[joint_idx] + amplitude * np.sin(omega * t)
        qdot[:, joint_idx] = amplitude * omega * np.cos(omega * t)
        qdotdot[:, joint_idx] = -amplitude * omega**2 * np.sin(omega * t)

        return {"t": t, "q": q, "qdot": qdot, "qdotdot": qdotdot}

    def multi_joint_smooth(
        self,
        duration: float,
        dt: float,
        n_waypoints: int = 6,
        seed: int = None,
    ) -> dict:
        """Random smooth multi-joint trajectory via cubic spline interpolation."""
        rng = np.random.default_rng(seed)

        # Waypoints at evenly spaced times; first and last at home
        t_wp = np.linspace(0.0, duration, n_waypoints)
        q_wp = np.zeros((n_waypoints, 6))
        q_wp[0] = HOME_POSE
        q_wp[-1] = HOME_POSE

        safe_frac = 0.5
        for j in range(6):
            lo = JOINT_LIMITS[j, 0] * safe_frac
            hi = JOINT_LIMITS[j, 1] * safe_frac
            q_wp[1:-1, j] = rng.uniform(lo, hi, n_waypoints - 2)

        cs = CubicSpline(t_wp, q_wp, bc_type="clamped")

        t = np.arange(0.0, duration, dt)
        q = cs(t)
        qdot = cs(t, 1)
        qdotdot = cs(t, 2)

        return {"t": t, "q": q, "qdot": qdot, "qdotdot": qdotdot}

    def low_amplitude(
        self,
        duration: float,
        dt: float,
        amplitude: float = 0.1,
    ) -> dict:
        """Small sine waves on all joints with different frequencies near home."""
        t = np.arange(0.0, duration, dt)
        n = len(t)
        frequencies = np.array([0.1, 0.15, 0.2, 0.12, 0.18, 0.25])

        q = np.zeros((n, 6))
        qdot = np.zeros((n, 6))
        qdotdot = np.zeros((n, 6))

        for j in range(6):
            omega = 2.0 * np.pi * frequencies[j]
            q[:, j] = HOME_POSE[j] + amplitude * np.sin(omega * t)
            qdot[:, j] = amplitude * omega * np.cos(omega * t)
            qdotdot[:, j] = -amplitude * omega**2 * np.sin(omega * t)

        return {"t": t, "q": q, "qdot": qdot, "qdotdot": qdotdot}
