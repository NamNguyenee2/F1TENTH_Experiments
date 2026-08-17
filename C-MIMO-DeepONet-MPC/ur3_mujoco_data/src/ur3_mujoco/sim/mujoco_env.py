import mujoco
import numpy as np
from .model_loader import load_model


class MujocoEnv:
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.model, self.data = load_model(xml_path)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_q_by_addr(self, qpos_addrs: list, q: np.ndarray):
        for addr, val in zip(qpos_addrs, q):
            self.data.qpos[addr] = float(val)

    def set_qdot_by_addr(self, qvel_addrs: list, qdot: np.ndarray):
        for addr, val in zip(qvel_addrs, qdot):
            self.data.qvel[addr] = float(val)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_site_pos(self, site_id: int) -> np.ndarray:
        return self.data.site_xpos[site_id].copy()

    def get_site_xmat(self, site_id: int) -> np.ndarray:
        return self.data.site_xmat[site_id].copy()

    def compute_jac_site(self, site_id: int):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return jacp, jacr
