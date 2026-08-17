import numpy as np


class PDController:
    """Position controller for MuJoCo position actuators.

    Since the MJCF uses position actuators with kp set in the model,
    setting ctrl = q_ref is sufficient. The model handles PD internally.
    """

    def __init__(self, model, kp: float = 500.0, kd: float = 50.0):
        self.model = model
        self.kp = kp
        self.kd = kd

    def compute(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        q_ref: np.ndarray,
        qdot_ref: np.ndarray,
        qdotdot_ref: np.ndarray,
    ) -> np.ndarray:
        """Return control command (desired joint positions for position actuators)."""
        return q_ref.copy()
