import time
import numpy as np
import h5py
import mujoco

from .model import UR3Model
from .controller import PDController
from .trajectory import TrajectoryGenerator
from .logger import DataLogger


class DatasetGenerator:
    """Generates and saves a dataset of UR3 trajectories to HDF5."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def generate(
        self,
        n_trajectories: int,
        duration: float,
        dt: float,
        traj_type: str,
        output_path: str,
        seed: int = 0,
    ):
        ur3 = UR3Model(self.model_path)
        ctrl = PDController(ur3)
        traj_gen = TrajectoryGenerator()
        logger = DataLogger(ur3)

        metadata = {
            "mujoco_version": mujoco.__version__,
            "model_path": self.model_path,
            "timestep": dt,
            "joint_names": ",".join(ur3.JOINT_NAMES),
            "ee_site": "ee_site",
            "traj_type": traj_type,
            "n_trajectories": n_trajectories,
            "duration": duration,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with h5py.File(output_path, "w") as f:
            meta_grp = f.create_group("metadata")
            for k, v in metadata.items():
                meta_grp.attrs[k] = str(v)

            for traj_i in range(n_trajectories):
                seed_i = seed + traj_i

                if traj_type == "sine":
                    joint_idx = traj_i % 6
                    traj = traj_gen.single_joint_sine(joint_idx=joint_idx, amplitude=0.4, frequency=0.2, duration=duration, dt=dt)
                elif traj_type == "multi":
                    traj = traj_gen.multi_joint_smooth(duration=duration, dt=dt, n_waypoints=8, seed=seed_i)
                else:
                    traj = traj_gen.low_amplitude(duration=duration, dt=dt)

                ur3.reset(traj["q"][0])
                logger.start_episode()

                has_nan = False
                for i in range(len(traj["t"])):
                    c = ctrl.compute(ur3.get_q(), ur3.get_qdot(), traj["q"][i], traj["qdot"][i], traj["qdotdot"][i])
                    ur3.set_ctrl(c)
                    ur3.step()
                    logger.log_step(ur3.data, c, traj["q"][i], traj["qdot"][i], traj["qdotdot"][i])

                ep = logger.get_episode_data()
                for k, v in ep.items():
                    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                        has_nan = True

                traj_grp = f.create_group(f"trajectory_{traj_i:03d}")
                for k, v in ep.items():
                    traj_grp.create_dataset(k, data=v)
                traj_grp.attrs["traj_type"] = traj_type
                traj_grp.attrs["seed"] = seed_i
                traj_grp.attrs["has_nan"] = has_nan

                if (traj_i + 1) % 10 == 0 or traj_i == 0:
                    print(f"  [{traj_i + 1}/{n_trajectories}] saved trajectory_{traj_i:03d}")

        return output_path
