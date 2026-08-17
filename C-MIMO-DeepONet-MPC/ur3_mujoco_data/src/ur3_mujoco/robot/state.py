import numpy as np
from dataclasses import dataclass


@dataclass
class RobotState:
    q: np.ndarray            # [6] radians
    q_dot: np.ndarray        # [6] radians/s
    ee_pos: np.ndarray       # [3] meters, world frame
    ee_quat: np.ndarray      # [4] [w,x,y,z], world frame
    ee_rotmat: np.ndarray    # [9] row-major, world frame
    ee_rot6d: np.ndarray     # [6] first 2 cols of R
    J_pos: np.ndarray        # [3,6] translational Jacobian, world frame
    J_rot: np.ndarray        # [3,6] rotational Jacobian, world frame
    ee_lin_vel: np.ndarray   # [3] m/s, world frame = J_pos @ q_dot
    ee_ang_vel: np.ndarray   # [3] rad/s, world frame = J_rot @ q_dot
