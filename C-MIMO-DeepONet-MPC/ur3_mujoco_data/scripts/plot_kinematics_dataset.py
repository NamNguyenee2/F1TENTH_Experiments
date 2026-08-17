#!/usr/bin/env python3
"""Generate validation plots for a kinematics dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
from pathlib import Path
from ur3_mujoco.validation.plots import plot_debug_trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or str(Path(args.dataset).parent / "plots")
    print(f"Generating plots → {out_dir}")
    plot_debug_trajectories(args.dataset, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
