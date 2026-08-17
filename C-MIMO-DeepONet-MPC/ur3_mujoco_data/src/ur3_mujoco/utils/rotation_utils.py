import numpy as np
from scipy.spatial.transform import Rotation


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    r = Rotation.from_matrix(R)
    xyzw = r.as_quat()  # scipy returns [x, y, z, w]
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R, flattened to shape [6]: [col0, col1]."""
    return np.concatenate([R[:, 0], R[:, 1]])


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Reconstruct orthonormal rotation matrix from 6D representation."""
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def check_rotation_matrix(R: np.ndarray, tol: float = 1e-5) -> tuple:
    """Returns (det_error, orth_error)."""
    det_err = float(abs(np.linalg.det(R) - 1.0))
    orth_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    return det_err, orth_err
