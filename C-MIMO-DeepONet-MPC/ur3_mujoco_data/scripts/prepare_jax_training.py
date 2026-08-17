#!/usr/bin/env python3
"""
Read UR3e kinematics dataset and prepare it for JAX training.

What this script does:
  1. Loads the HDF5 dataset
  2. Fits a normalizer on the train split
  3. Saves normalizer stats as JSON
  4. Runs a quick sanity check on the data
  5. Shows how to iterate batches (drop-in for your training loop)

Usage:
  python scripts/prepare_jax_training.py --dataset data/small/ur3_kinematics_small.h5
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from ur3_mujoco.training.jax_dataloader import KinematicsDataset

BASE_DIR = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/small/ur3_kinematics_small.h5')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--normalize', action='store_true', default=True)
    args = parser.parse_args()

    dataset_path = str(BASE_DIR / args.dataset)
    norm_path    = str(Path(dataset_path).parent / "normalizer.json")

    # ── 1. Load dataset ─────────────────────────────────────────────
    print("=" * 60)
    print("UR3e Kinematics — JAX Data Loader")
    print("=" * 60)
    ds = KinematicsDataset(dataset_path, normalize=args.normalize)

    # ── 2. Fit and save normalizer ───────────────────────────────────
    print()
    normalizer = ds.fit_normalizer(split="train")
    ds.save_normalizer(norm_path)

    # ── 3. Sanity checks ─────────────────────────────────────────────
    print("\n[Sanity checks]")

    # Single batch from train
    x_batch, y_batch = next(ds.batches("train", batch_size=args.batch_size))
    print(f"  train batch   x: {x_batch.shape}  y: {y_batch.shape}  dtype: {x_batch.dtype}")

    if args.normalize:
        print(f"  x_batch  mean={float(x_batch.mean()):.4f}  std={float(x_batch.std()):.4f}  (should be ~0, ~1)")
        print(f"  y_batch  mean={float(y_batch.mean()):.4f}  std={float(y_batch.std()):.4f}  (should be ~0, ~1)")

    # Check val and test splits
    x_val, y_val = ds.get_split("val")
    print(f"  val   x: {x_val.shape}  y: {y_val.shape}")

    x_test, y_test = ds.get_split("test_random")
    print(f"  test_random   x: {x_test.shape}  y: {y_test.shape}")

    if ds.size("test_stress") > 0:
        x_stress, y_stress = ds.get_split("test_stress")
        print(f"  test_stress   x: {x_stress.shape}  y: {y_stress.shape}  (high-speed OOD)")

    # ── 4. Input/output breakdown ────────────────────────────────────
    print("\n[Input / Output breakdown]")
    print("  x [N, 18] = [ sin(q)(6) | cos(q)(6) | q_dot(6) ]")
    print("  y [N, 15] = [ ee_pos(3) | ee_rot6d(6) | ee_lin_vel(3) | ee_ang_vel(3) ]")

    print("\n[Denormalization example]")
    y_pred_norm = jnp.zeros((4, 15))   # dummy model output (normalized)
    y_pred = normalizer.denormalize_y(np.array(y_pred_norm))
    ee_pos = y_pred[:, :3]
    ee_rot6d = y_pred[:, 3:9]
    ee_lin_vel = y_pred[:, 9:12]
    ee_ang_vel = y_pred[:, 12:15]
    print(f"  Denormalized ee_pos  : {ee_pos[0]}  [m]")
    print(f"  Denormalized lin_vel : {ee_lin_vel[0]}  [m/s]")

    # ── 5. Batch iteration demo ──────────────────────────────────────
    print("\n[Batch iteration demo — 1 epoch over train]")
    n_batches = 0
    for x, y in ds.batches("train", batch_size=args.batch_size, shuffle=True, rng_seed=42):
        n_batches += 1
    n_train = ds.size("train")
    print(f"  batch_size={args.batch_size}  "
          f"n_train={n_train:,}  "
          f"batches_per_epoch={n_batches}")

    # ── 6. Summary for copy-paste ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Copy-paste snippet for your training loop:")
    print("=" * 60)
    print(f"""
from ur3_mujoco.training.jax_dataloader import KinematicsDataset, Normalizer

ds = KinematicsDataset("{dataset_path}", normalize=True)
ds.load_normalizer("{norm_path}")

x_val, y_val = ds.get_split("val")       # full val set, jax arrays

for epoch in range(NUM_EPOCHS):
    for x, y in ds.batches("train", batch_size=BATCH_SIZE,
                            shuffle=True, rng_seed=epoch):
        # x: jax array [B, 18]  — normalised inputs
        # y: jax array [B, 15]  — normalised targets
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        params = update(params, grads)

# Evaluate on test splits
x_test, y_test = ds.get_split("test_random")
x_ood,  y_ood  = ds.get_split("test_stress")   # high-speed OOD

# Denormalize predictions
y_pred_real = ds.normalizer.denormalize_y(np.array(y_pred_norm))
ee_pos      = y_pred_real[:, :3]    # [B, 3]  metres
ee_rot6d    = y_pred_real[:, 3:9]   # [B, 6]  reconstruct R via Gram-Schmidt
ee_lin_vel  = y_pred_real[:, 9:12]  # [B, 3]  m/s
ee_ang_vel  = y_pred_real[:, 12:15] # [B, 3]  rad/s
""")
    print(f"Normalizer saved to: {norm_path}")
    print("Done.")


if __name__ == "__main__":
    main()
