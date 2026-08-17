import numpy as np


class Validator:
    """Validation utilities for UR3 simulation data."""

    @staticmethod
    def check_rotation_matrix(R: np.ndarray) -> dict:
        """Verify det≈1 and RᵀR≈I for a (3,3) rotation matrix."""
        det_err = abs(np.linalg.det(R) - 1.0)
        orth_err = np.max(np.abs(R.T @ R - np.eye(3)))
        return {"det_error": det_err, "orthogonality_error": orth_err, "valid": det_err < 1e-4 and orth_err < 1e-4}

    @staticmethod
    def check_no_nan_inf(data_dict: dict) -> dict:
        results = {}
        for k, v in data_dict.items():
            arr = np.asarray(v)
            results[k] = {"has_nan": bool(np.any(np.isnan(arr))), "has_inf": bool(np.any(np.isinf(arr)))}
        return results

    @staticmethod
    def check_shapes(data_dict: dict) -> dict:
        expected = {
            "time": (-1,),
            "q": (-1, 6),
            "q_dot": (-1, 6),
            "q_dot_dot": (-1, 6),
            "ee_pos": (-1, 3),
            "ee_quat": (-1, 4),
            "ee_rotmat": (-1, 9),
            "ee_lin_vel": (-1, 3),
            "ee_ang_vel": (-1, 3),
            "ctrl": (-1, 6),
        }
        results = {}
        for k, exp_shape in expected.items():
            if k not in data_dict:
                results[k] = "MISSING"
                continue
            actual = np.asarray(data_dict[k]).shape
            match = all(e == -1 or e == a for e, a in zip(exp_shape, actual))
            results[k] = {"actual": actual, "expected": exp_shape, "match": match}
        return results

    @staticmethod
    def velocity_consistency(ee_pos: np.ndarray, ee_lin_vel: np.ndarray, dt: float) -> float:
        """Compare Jacobian-based velocity to finite-difference of ee_pos."""
        ee_vel_fd = np.gradient(ee_pos, dt, axis=0)
        return float(np.max(np.abs(ee_lin_vel - ee_vel_fd)))

    @staticmethod
    def acceleration_consistency(qdot: np.ndarray, qdotdot: np.ndarray, dt: float) -> float:
        """Compare logged qacc to finite-difference of qdot."""
        qdotdot_fd = np.gradient(qdot, dt, axis=0)
        return float(np.max(np.abs(qdotdot - qdotdot_fd)))
