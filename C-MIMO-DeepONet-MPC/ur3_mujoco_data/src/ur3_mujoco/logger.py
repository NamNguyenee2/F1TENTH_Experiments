import numpy as np
import h5py


class DataLogger:
    """Records simulation data for each episode and persists to HDF5."""

    _FIELDS = [
        "time", "q", "q_dot", "q_dot_dot",
        "ee_pos", "ee_quat", "ee_rotmat",
        "ee_lin_vel", "ee_ang_vel",
        "ctrl", "q_ref", "q_dot_ref", "q_dot_dot_ref",
    ]

    def __init__(self, model):
        self.model = model
        self._buffers = {f: [] for f in self._FIELDS}

    def start_episode(self):
        self._buffers = {f: [] for f in self._FIELDS}

    def log_step(self, data, ctrl, q_ref, qdot_ref, qdotdot_ref):
        m = self.model

        q = m.get_q()
        qdot = m.get_qdot()
        qdotdot = m.get_qdotdot()
        ee_pos = m.get_ee_pos()
        ee_quat = m.get_ee_quat()
        ee_rotmat = m.get_ee_rotmat()

        J_lin, J_rot = m.get_ee_vel_jacobian(data)
        ee_lin_vel = J_lin @ qdot
        ee_ang_vel = J_rot @ qdot

        self._buffers["time"].append(float(data.time))
        self._buffers["q"].append(q.copy())
        self._buffers["q_dot"].append(qdot.copy())
        self._buffers["q_dot_dot"].append(qdotdot.copy())
        self._buffers["ee_pos"].append(ee_pos.copy())
        self._buffers["ee_quat"].append(ee_quat.copy())
        self._buffers["ee_rotmat"].append(ee_rotmat.copy())
        self._buffers["ee_lin_vel"].append(ee_lin_vel.copy())
        self._buffers["ee_ang_vel"].append(ee_ang_vel.copy())
        self._buffers["ctrl"].append(np.asarray(ctrl).copy())
        self._buffers["q_ref"].append(np.asarray(q_ref).copy())
        self._buffers["q_dot_ref"].append(np.asarray(qdot_ref).copy())
        self._buffers["q_dot_dot_ref"].append(np.asarray(qdotdot_ref).copy())

    def get_episode_data(self) -> dict:
        return {k: np.array(v) for k, v in self._buffers.items()}

    def save_to_hdf5(self, filepath: str, traj_name: str, metadata: dict):
        ep = self.get_episode_data()
        with h5py.File(filepath, "a") as f:
            grp = f.require_group(traj_name)
            for k, v in ep.items():
                if k in grp:
                    del grp[k]
                grp.create_dataset(k, data=v)
            for k, v in metadata.items():
                grp.attrs[k] = str(v)

    @staticmethod
    def load_from_hdf5(filepath: str, traj_name: str) -> dict:
        result = {}
        with h5py.File(filepath, "r") as f:
            grp = f[traj_name]
            for k in grp.keys():
                result[k] = grp[k][:]
        return result
