import mujoco
import numpy as np
from ..sim.mujoco_env import MujocoEnv
from .state import RobotState
from .kinematics import compute_ee_state

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class UR3Robot:
    """High-level UR3e interface. All joint access is by name, never raw index."""

    def __init__(self, env: MujocoEnv, ee_site_name: str = "ee_site"):
        self.env = env
        model = env.model

        self._qpos_addrs = []
        self._qvel_addrs = []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in model")
            self._qpos_addrs.append(int(model.jnt_qposadr[jid]))
            self._qvel_addrs.append(int(model.jnt_dofadr[jid]))

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if site_id < 0:
            raise ValueError(f"Site '{ee_site_name}' not found in model")
        self._ee_site_id = site_id

    def set_state(self, q: np.ndarray, qdot: np.ndarray = None):
        """Set joint positions (and optionally velocities), then run mj_forward."""
        self.env.set_q_by_addr(self._qpos_addrs, q)
        qdot_val = qdot if qdot is not None else np.zeros(6)
        self.env.set_qdot_by_addr(self._qvel_addrs, qdot_val)
        self.env.forward()

    def get_q(self) -> np.ndarray:
        return np.array([self.env.data.qpos[a] for a in self._qpos_addrs])

    def get_qdot(self) -> np.ndarray:
        return np.array([self.env.data.qvel[a] for a in self._qvel_addrs])

    def get_state(self, q_dot: np.ndarray = None) -> RobotState:
        q = self.get_q()
        qdot = q_dot if q_dot is not None else self.get_qdot()
        ee_pos, ee_quat, ee_rotmat, ee_rot6d, J_pos, J_rot, ee_lin_vel, ee_ang_vel = \
            compute_ee_state(self.env, self._ee_site_id, self._qvel_addrs, qdot)
        return RobotState(
            q=q, q_dot=qdot,
            ee_pos=ee_pos, ee_quat=ee_quat,
            ee_rotmat=ee_rotmat, ee_rot6d=ee_rot6d,
            J_pos=J_pos, J_rot=J_rot,
            ee_lin_vel=ee_lin_vel, ee_ang_vel=ee_ang_vel,
        )
