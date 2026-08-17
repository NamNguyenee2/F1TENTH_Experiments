#!/usr/bin/env python3
"""Inspect the structure and shapes of a kinematics HDF5 dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    with h5py.File(args.dataset, 'r') as f:
        print("=" * 60)
        print(f"Dataset: {args.dataset}")
        print("=" * 60)

        if "metadata" in f:
            print("\n[Metadata]")
            for k, v in f["metadata"].attrs.items():
                print(f"  {k}: {v}")

        for split in ["train", "val", "test_random", "test_trajectory", "test_boundary"]:
            if split in f:
                print(f"\n[/{split}]  ({f[split]['q'].shape[0]} samples)")
                for k in sorted(f[split].keys()):
                    ds = f[split][k]
                    print(f"  {k:25s} {str(ds.shape):20s} {ds.dtype}")

        if "debug_trajectories" in f:
            print("\n[/debug_trajectories]")
            for subgrp in f["debug_trajectories"].keys():
                n = len(f["debug_trajectories"][subgrp].keys())
                print(f"  {subgrp}: {n} trajectories")
                if n > 0:
                    key0 = sorted(f["debug_trajectories"][subgrp].keys())[0]
                    for k in sorted(f["debug_trajectories"][subgrp][key0].keys()):
                        ds = f["debug_trajectories"][subgrp][key0][k]
                        print(f"    {k:25s} {str(ds.shape)}")


if __name__ == "__main__":
    main()
