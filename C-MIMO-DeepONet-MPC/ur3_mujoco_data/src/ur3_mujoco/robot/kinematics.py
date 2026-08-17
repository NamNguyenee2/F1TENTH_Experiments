import numpy as np
from ..utils.rotation_utils import rotmat_to_quat_wxyz, rotmat_to_rot6d


def compute_ee_state(env, ee_site_id: int, qvel_idxs: list, q_dot: np.ndarray):
    """Compute full EE state from MuJoCo forward pass. All quantities in world frame."""
    ee_pos = env.get_site_pos(ee_site_id)
    xmat = env.get_site_xmat(ee_site_id)   # shape (9,) row-major
    R = xmat.reshape(3, 3)
    ee_rotmat = xmat.copy()
    ee_quat = rotmat_to_quat_wxyz(R)
    ee_rot6d = rotmat_to_rot6d(R)

    jacp, jacr = env.compute_jac_site(ee_site_id)
    J_pos = jacp[:, qvel_idxs].copy()      # [3, 6]
    J_rot = jacr[:, qvel_idxs].copy()      # [3, 6]

    ee_lin_vel = J_pos @ q_dot
    ee_ang_vel = J_rot @ q_dot

    return ee_pos, ee_quat, ee_rotmat, ee_rot6d, J_pos, J_rot, ee_lin_vel, ee_ang_vel
