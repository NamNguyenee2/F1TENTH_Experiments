import numpy as np


def sample_q(rng: np.random.Generator, joint_limits: np.ndarray,
             safe_fraction: float = 0.7) -> np.ndarray:
    lo = joint_limits[:, 0] * safe_fraction
    hi = joint_limits[:, 1] * safe_fraction
    return rng.uniform(lo, hi)


def sample_qdot(rng: np.random.Generator, vel_limits) -> np.ndarray:
    lo = np.array([v[0] for v in vel_limits], dtype=float)
    hi = np.array([v[1] for v in vel_limits], dtype=float)
    return rng.uniform(lo, hi)
