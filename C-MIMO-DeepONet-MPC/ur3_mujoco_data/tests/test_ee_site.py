"""Test: ee_site exists and returns valid position and orientation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import mujoco
import numpy as np
from pathlib import Path
from ur3_mujoco.utils.rotation_utils import check_rotation_matrix

MODEL_PATH = Path(__file__).parent.parent / "models" / "scene.xml"
EE_SITE_NAME = "ee_site"


def test_ee_site_exists():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    assert site_id >= 0, f"Site '{EE_SITE_NAME}' not found"
    print(f"PASS test_ee_site_exists (id={site_id})")


def test_ee_site_pos_valid():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    pos = data.site_xpos[site_id]
    assert pos.shape == (3,), f"pos shape {pos.shape}"
    assert not np.any(np.isnan(pos)), "pos contains NaN"
    assert not np.any(np.isinf(pos)), "pos contains Inf"
    print(f"PASS test_ee_site_pos_valid: pos={pos}")


def test_ee_site_rotmat_valid():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    xmat = data.site_xmat[site_id].reshape(3, 3)
    det_err, orth_err = check_rotation_matrix(xmat)
    assert det_err < 1e-5, f"det(R) error too large: {det_err}"
    assert orth_err < 1e-5, f"R.T@R error too large: {orth_err}"
    print(f"PASS test_ee_site_rotmat_valid (det_err={det_err:.2e}, orth_err={orth_err:.2e})")


if __name__ == "__main__":
    test_ee_site_exists()
    test_ee_site_pos_valid()
    test_ee_site_rotmat_valid()
    print("\nAll ee_site tests PASS")
