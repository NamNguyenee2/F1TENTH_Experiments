import numpy as np

HOME_POSE = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


def generate(amplitudes_rad: np.ndarray, frequencies: np.ndarray,
             phases: np.ndarray, duration: float, dt: float) -> dict:
    """Multi-joint sinusoidal trajectory, all joints moving simultaneously."""
    t = np.arange(0.0, duration, dt)
    n = len(t)
    q = np.zeros((n, 6))
    qdot = np.zeros((n, 6))
    qdotdot = np.zeros((n, 6))
    for j in range(6):
        omega = 2.0 * np.pi * frequencies[j]
        q[:, j] = HOME_POSE[j] + amplitudes_rad[j] * np.sin(omega * t + phases[j])
        qdot[:, j] = amplitudes_rad[j] * omega * np.cos(omega * t + phases[j])
        qdotdot[:, j] = -amplitudes_rad[j] * omega ** 2 * np.sin(omega * t + phases[j])
    return {"t": t, "q": q, "qdot": qdot, "qdotdot": qdotdot}
