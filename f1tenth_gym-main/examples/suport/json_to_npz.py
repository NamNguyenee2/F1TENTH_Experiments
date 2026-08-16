"""
Converts a run log JSON (e.g. map_oschersleben_wide_lap3.json, written at the
end of waypoint_follow.py's main()) into a compressed .npz archive.

Usage:
    python json_to_npz.py map_oschersleben_wide_lap3.json [more.json ...]

With no arguments, converts every map_*.json file in the current directory.
Each top-level JSON key becomes an array in the .npz, loadable via:
    data = np.load('map_oschersleben_wide_lap3.npz')
    data['x'], data['y'], data['yaw'], ...
"""
import glob
import json
import os
import sys

import numpy as np


def convert(json_path):
    with open(json_path) as f:
        log = json.load(f)

    arrays = {key: np.asarray(val) for key, val in log.items()}
    npz_path = os.path.splitext(json_path)[0] + '.npz'
    np.savez_compressed(npz_path, **arrays)

    print(f'{json_path} -> {npz_path}')
    for key, arr in arrays.items():
        print(f'  {key}: shape={arr.shape} dtype={arr.dtype}')


def main():
    paths = sys.argv[1:] or glob.glob('map_*.json')
    if not paths:
        print('No JSON log files found (expected map_*.json in the current directory).')
        return
    for path in paths:
        convert(path)


if __name__ == '__main__':
    main()
