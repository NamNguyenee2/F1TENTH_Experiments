import h5py
import numpy as np
from ..utils.rotation_utils import check_rotation_matrix, rot6d_to_rotmat
from ..dataset.kinematics_schema import SPLIT_FIELDS, FIELD_SHAPES

REQUIRED_SPLITS = ["train", "val", "test_random"]


class KinematicsDatasetValidator:
    def __init__(self, dataset_path: str, config: dict):
        self.path = dataset_path
        self.config = config
        self.errors = []
        self.warnings = []

    def _err(self, msg):
        self.errors.append(msg)
        print(f"  ERROR: {msg}")

    def _warn(self, msg):
        self.warnings.append(msg)
        print(f"  WARN:  {msg}")

    def _ok(self, msg):
        print(f"  OK:    {msg}")

    def validate(self) -> bool:
        print(f"\n{'='*60}")
        print(f"Validating: {self.path}")
        print(f"{'='*60}")
        with h5py.File(self.path, 'r') as f:
            self._check_metadata(f)
            for split in REQUIRED_SPLITS:
                if split in f:
                    self._check_split(f, split)
                else:
                    self._warn(f"split '{split}' missing")
            self._check_trajectories(f)
        ok = len(self.errors) == 0
        print(f"\nResult: {'PASS' if ok else 'FAIL'} "
              f"({len(self.errors)} errors, {len(self.warnings)} warnings)")
        return ok

    def _check_metadata(self, f):
        print("\n[Metadata]")
        required = ["robot_name", "mujoco_version", "joint_names",
                    "ee_site_name", "date_generated"]
        if "metadata" not in f:
            self._err("metadata group missing")
            return
        for k in required:
            if k not in f["metadata"].attrs:
                self._err(f"metadata.{k} missing")
            else:
                self._ok(f"metadata.{k} = {f['metadata'].attrs[k]}")

    def _check_split(self, f, split_name):
        print(f"\n[Split: {split_name}]")
        grp = f[split_name]
        N = None

        for field in SPLIT_FIELDS:
            if field not in grp:
                self._err(f"{split_name}/{field} missing")
                continue
            arr = grp[field][:]
            if N is None:
                N = arr.shape[0]

            expected = FIELD_SHAPES[field]
            if len(arr.shape) != len(expected):
                self._err(f"{split_name}/{field}: ndim {arr.shape} != {expected}")
            else:
                shape_ok = all(a == e or e == -1 for a, e in zip(arr.shape, expected))
                if not shape_ok:
                    self._err(f"{split_name}/{field}: shape {arr.shape} != {expected}")

            if np.any(np.isnan(arr)):
                self._err(f"{split_name}/{field}: contains NaN")
            if np.any(np.isinf(arr)):
                self._err(f"{split_name}/{field}: contains Inf")

        if N is None or N == 0:
            self._err(f"{split_name}: empty")
            return
        self._ok(f"{split_name}: N={N} samples")

        # Joint limits
        jl = np.array(self.config['robot']['joint_limits'])
        q = grp["q"][:]
        for j in range(6):
            lo, hi = jl[j]
            if np.any(q[:, j] < lo - 1e-6) or np.any(q[:, j] > hi + 1e-6):
                self._err(f"{split_name}/q joint {j} out of limits")
        self._ok(f"{split_name}/q within joint limits")

        # Quaternion normalization
        quat = grp["ee_quat_world"][:]
        norms = np.linalg.norm(quat, axis=1)
        max_quat_err = float(np.max(np.abs(norms - 1.0)))
        if max_quat_err > 1e-4:
            self._err(f"{split_name}/ee_quat_world not normalized, max_err={max_quat_err:.2e}")
        else:
            self._ok(f"{split_name}/ee_quat_world normalized (max_err={max_quat_err:.2e})")

        # Rotation matrix validity (random sample)
        rm = grp["ee_rotmat_world"][:]
        n_check = min(200, N)
        idx = np.random.default_rng(0).choice(N, n_check, replace=False)
        max_det_err = max_orth_err = 0.0
        for i in idx:
            R = rm[i].reshape(3, 3)
            d, o = check_rotation_matrix(R)
            max_det_err = max(max_det_err, d)
            max_orth_err = max(max_orth_err, o)
        if max_det_err > 1e-4:
            self._err(f"{split_name}/ee_rotmat_world invalid: det_err={max_det_err:.2e}")
        else:
            self._ok(f"{split_name}/ee_rotmat_world valid (det_err={max_det_err:.2e}, orth_err={max_orth_err:.2e})")

        # rot6d reconstruction
        rot6d = grp["ee_rot6d_world"][:]
        for i in idx[:20]:
            R_rec = rot6d_to_rotmat(rot6d[i])
            d, o = check_rotation_matrix(R_rec)
            if d > 1e-4 or o > 1e-4:
                self._err(f"{split_name}/ee_rot6d_world reconstruction failed at {i}")
                break
        else:
            self._ok(f"{split_name}/ee_rot6d_world reconstruction valid")

        # Velocity consistency: ee_lin_vel = J_pos @ q_dot
        J_pos = grp["J_pos_world"][:]
        J_rot = grp["J_rot_world"][:]
        qdot = grp["q_dot"][:]
        lin_vel = grp["ee_lin_vel_world"][:]
        ang_vel = grp["ee_ang_vel_world"][:]
        max_vel_err = 0.0
        for i in idx[:100]:
            lv = J_pos[i] @ qdot[i]
            av = J_rot[i] @ qdot[i]
            max_vel_err = max(max_vel_err,
                              float(np.max(np.abs(lin_vel[i] - lv))),
                              float(np.max(np.abs(ang_vel[i] - av))))
        if max_vel_err > 1e-7:
            self._err(f"{split_name}: vel != J@qdot, max_err={max_vel_err:.2e}")
        else:
            self._ok(f"{split_name}: ee_lin/ang_vel = J@q_dot (max_err={max_vel_err:.2e})")

        # y content check
        y = grp["y"][:]
        ee_pos = grp["ee_pos_world"][:]
        rot6d = grp["ee_rot6d_world"][:]
        if not np.allclose(y[:, :3], ee_pos, atol=1e-8):
            self._err(f"{split_name}/y[:,:3] != ee_pos_world")
        elif not np.allclose(y[:, 3:9], rot6d, atol=1e-8):
            self._err(f"{split_name}/y[:,3:9] != ee_rot6d_world")
        else:
            self._ok(f"{split_name}/y consistent with ee_pos and ee_rot6d")

    def _check_trajectories(self, f):
        for path in ["debug_trajectories/single_joint_sine",
                     "debug_trajectories/multi_joint_sine"]:
            if path in f:
                n = len(f[path].keys())
                self._ok(f"{path}: {n} trajectories")
            else:
                self._warn(f"{path}: not found")
