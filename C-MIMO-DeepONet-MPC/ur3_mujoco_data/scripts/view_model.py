#!/usr/bin/env python3
"""Launch MuJoCo viewer for UR3 model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import mujoco
import mujoco.viewer
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "scene.xml"

def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    print("Launching viewer. Close window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        import time
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)

if __name__ == "__main__":
    main()
