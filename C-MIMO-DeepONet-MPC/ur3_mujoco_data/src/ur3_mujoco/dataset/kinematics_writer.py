import h5py
import numpy as np
import time
import mujoco
from pathlib import Path
from ..utils.file_utils import md5_file


def _write_split(grp: h5py.Group, data: dict, chunk_size: int = 1000):
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.ndim == 0:
            grp.attrs[k] = v
            continue
        chunks = (min(chunk_size, arr.shape[0]),) + arr.shape[1:]
        if k in grp:
            del grp[k]
        grp.create_dataset(k, data=arr, chunks=chunks,
                           compression="gzip", compression_opts=4)


def write_dataset(output_path: str, splits: dict, traj_groups: dict,
                  config: dict, model_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model_hash = md5_file(model_path)

    with h5py.File(output_path, 'w') as f:
        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["robot_name"] = "UR3e"
        meta.attrs["mujoco_version"] = mujoco.__version__
        meta.attrs["model_path"] = model_path
        meta.attrs["model_hash"] = model_hash
        meta.attrs["joint_names"] = ",".join(config['robot']['joint_names'])
        meta.attrs["joint_limits"] = str(config['robot']['joint_limits'])
        meta.attrs["q_velocity_limits"] = str(config['robot']['q_velocity_limits'])
        meta.attrs["ee_site_name"] = config['robot']['ee_site_name']
        meta.attrs["ee_site_definition"] = "flange/tool0 frame on wrist_3_link"
        meta.attrs["pose_frame"] = "world"
        meta.attrs["velocity_frame"] = "world"
        meta.attrs["orientation_convention"] = "quat=[w,x,y,z]; rot6d=first 2 cols of R"
        meta.attrs["sampling_strategy"] = config['dataset']['stage']
        meta.attrs["date_generated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Data splits
        for split_name, data in splits.items():
            if data is not None and len(data) > 0:
                grp = f.require_group(split_name)
                _write_split(grp, data)

        # Trajectory groups (nested: outer/inner/traj_NNN/fields)
        for outer_name, inner_dict in traj_groups.items():
            outer_grp = f.require_group(outer_name)
            for inner_name, traj_list in inner_dict.items():
                inner_grp = outer_grp.require_group(inner_name)
                for i, traj in enumerate(traj_list):
                    tg = inner_grp.create_group(f"traj_{i:03d}")
                    for k, v in traj.items():
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            tg.attrs[k] = v
                        else:
                            tg.create_dataset(k, data=np.asarray(v))

    print(f"Dataset written: {output_path}")
