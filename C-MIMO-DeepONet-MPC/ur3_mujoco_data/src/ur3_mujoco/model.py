import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class UR3Model:
    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    ACTUATOR_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]

    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache joint ids and addresses
        self._joint_ids = {}
        self._qpos_addrs = {}
        self._qvel_addrs = {}
        for name in self.JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in model")
            self._joint_ids[name] = jid
            self._qpos_addrs[name] = self.model.jnt_qposadr[jid]
            self._qvel_addrs[name] = self.model.jnt_dofadr[jid]

        # Cache actuator ids
        self._actuator_ids = {}
        for name in self.ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Actuator '{name}' not found in model")
            self._actuator_ids[name] = aid

        # Cache ee_site id
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if self._ee_site_id < 0:
            raise ValueError("Site 'ee_site' not found in model")

        # Pre-allocate Jacobian arrays
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        # Qvel column indices for the 6 UR3 DOFs
        self._qvel_idxs = [self._qvel_addrs[n] for n in self.JOINT_NAMES]

    # ----- address lookups -----

    def joint_name_to_id(self, name: str) -> int:
        return self._joint_ids[name]

    def joint_name_to_qpos_addr(self, name: str) -> int:
        return self._qpos_addrs[name]

    def joint_name_to_qvel_addr(self, name: str) -> int:
        return self._qvel_addrs[name]

    # ----- state getters -----

    def get_q(self) -> np.ndarray:
        return np.array([self.data.qpos[self._qpos_addrs[n]] for n in self.JOINT_NAMES])

    def get_qdot(self) -> np.ndarray:
        return np.array([self.data.qvel[self._qvel_addrs[n]] for n in self.JOINT_NAMES])

    def get_qdotdot(self) -> np.ndarray:
        return np.array([self.data.qacc[self._qvel_addrs[n]] for n in self.JOINT_NAMES])

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site_id].copy()

    def get_ee_rotmat(self) -> np.ndarray:
        return self.data.site_xmat[self._ee_site_id].copy()  # shape (9,) row-major

    def get_ee_quat(self) -> np.ndarray:
        """Return quaternion [w, x, y, z]."""
        R = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
        r = Rotation.from_matrix(R)
        xyzw = r.as_quat()  # scipy returns [x, y, z, w]
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # [w, x, y, z]

    def get_ee_vel_jacobian(self, data=None):
        """Return (J_lin [3,6], J_rot [3,6]) sliced to UR3 DOFs."""
        if data is None:
            data = self.data
        self._jacp[:] = 0.0
        self._jacr[:] = 0.0
        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self._ee_site_id)
        J_lin = self._jacp[:, self._qvel_idxs]
        J_rot = self._jacr[:, self._qvel_idxs]
        return J_lin.copy(), J_rot.copy()

    # ----- control -----

    def set_ctrl(self, ctrl_6dof: np.ndarray):
        for i, name in enumerate(self.ACTUATOR_NAMES):
            self.data.ctrl[self._actuator_ids[name]] = ctrl_6dof[i]

    # ----- sim control -----

    def reset(self, q_init=None):
        mujoco.mj_resetData(self.model, self.data)
        if q_init is not None:
            for i, name in enumerate(self.JOINT_NAMES):
                self.data.qpos[self._qpos_addrs[name]] = q_init[i]
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)
