import numpy as np
from ..robot.ur3_robot import UR3Robot


def sample_one(robot: UR3Robot, q: np.ndarray, q_dot: np.ndarray) -> dict:
    """Set MuJoCo state, run mj_forward, extract all supervised labels."""
    robot.set_state(q, q_dot)
    state = robot.get_state(q_dot=q_dot)

    sin_q = np.sin(q)
    cos_q = np.cos(q)
    x = np.concatenate([sin_q, cos_q, q_dot])          # [18]
    J_world = np.vstack([state.J_pos, state.J_rot])     # [6, 6]
    y = np.concatenate([
        state.ee_pos,       # [3]
        state.ee_rot6d,     # [6]
        state.ee_lin_vel,   # [3]
        state.ee_ang_vel,   # [3]
    ])                                                   # [15]

    return {
        "q":                q.copy(),
        "q_dot":            q_dot.copy(),
        "sin_q":            sin_q,
        "cos_q":            cos_q,
        "x":                x,
        "ee_pos_world":     state.ee_pos,
        "ee_quat_world":    state.ee_quat,
        "ee_rotmat_world":  state.ee_rotmat,
        "ee_rot6d_world":   state.ee_rot6d,
        "ee_lin_vel_world": state.ee_lin_vel,
        "ee_ang_vel_world": state.ee_ang_vel,
        "J_pos_world":      state.J_pos,
        "J_rot_world":      state.J_rot,
        "J_world":          J_world,
        "y":                y,
    }
