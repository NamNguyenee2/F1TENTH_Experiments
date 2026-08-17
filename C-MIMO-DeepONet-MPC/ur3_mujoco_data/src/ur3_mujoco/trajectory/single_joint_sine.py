import numpy as np

HOME_POSE = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


def generate(joint_idx: int, amplitude_rad: float, frequency: float,
             duration: float, dt: float) -> dict:
    """Sinusoidal trajectory for one joint; all others fixed at home pose."""
    t = np.arange(0.0, duration, dt)
    n = len(t)
    omega = 2.0 * np.pi * frequency
    q = np.tile(HOME_POSE, (n, 1))
    qdot = np.zeros((n, 6))
    qdotdot = np.zeros((n, 6))
    q[:, joint_idx] = HOME_POSE[joint_idx] + amplitude_rad * np.sin(omega * t)
    qdot[:, joint_idx] = amplitude_rad * omega * np.cos(omega * t)
    qdotdot[:, joint_idx] = -amplitude_rad * omega ** 2 * np.sin(omega * t)
    return {"t": t, "q": q, "qdot": qdot, "qdotdot": qdotdot}
