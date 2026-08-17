# Field names for each split
SPLIT_FIELDS = [
    "q", "q_dot", "sin_q", "cos_q", "x",
    "ee_pos_world", "ee_quat_world", "ee_rotmat_world",
    "ee_rot6d_world", "ee_lin_vel_world", "ee_ang_vel_world",
    "J_pos_world", "J_rot_world", "J_world", "y",
]

# Expected shapes (-1 = any size)
FIELD_SHAPES = {
    "q":                (-1, 6),
    "q_dot":            (-1, 6),
    "sin_q":            (-1, 6),
    "cos_q":            (-1, 6),
    "x":                (-1, 18),
    "ee_pos_world":     (-1, 3),
    "ee_quat_world":    (-1, 4),
    "ee_rotmat_world":  (-1, 9),
    "ee_rot6d_world":   (-1, 6),
    "ee_lin_vel_world": (-1, 3),
    "ee_ang_vel_world": (-1, 3),
    "J_pos_world":      (-1, 3, 6),
    "J_rot_world":      (-1, 3, 6),
    "J_world":          (-1, 6, 6),
    "y":                (-1, 15),
}

TRAJECTORY_FIELDS = [
    "time", "q", "q_dot", "q_dot_dot",
    "ee_pos_world", "ee_quat_world", "ee_rotmat_world",
    "ee_rot6d_world", "ee_lin_vel_world", "ee_ang_vel_world", "J_world",
]
