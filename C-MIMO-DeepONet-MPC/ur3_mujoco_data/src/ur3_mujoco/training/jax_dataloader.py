"""JAX-compatible data loader for the UR3e kinematics HDF5 dataset.

Usage
-----
from ur3_mujoco.training.jax_dataloader import KinematicsDataset

ds = KinematicsDataset("data/small/ur3_kinematics_small.h5")
ds.fit_normalizer()          # compute mean/std from train split

# Iterate batches (returns jax arrays)
for x, y in ds.batches("train", batch_size=256, shuffle=True, rng_seed=0):
    # x: [B, 18]  = [sin(q), cos(q), q_dot]
    # y: [B, 15]  = [ee_pos(3), ee_rot6d(6), ee_lin_vel(3), ee_ang_vel(3)]
    ...

# Get a full split as a single jax array pair
x_val, y_val = ds.get_split("val")
"""

import h5py
import numpy as np
import json
from pathlib import Path
from typing import Iterator, Tuple, Optional

import jax
import jax.numpy as jnp


# ── Field config ────────────────────────────────────────────────────────────
X_FIELD = "x"          # [N, 18]  input:  [sin(q)(6), cos(q)(6), q_dot(6)]
Y_FIELD = "y"          # [N, 15]  output: [ee_pos(3), ee_rot6d(6), ee_lin_vel(3), ee_ang_vel(3)]

SPLITS = ["train", "val", "test_random", "test_trajectory", "test_stress"]


class Normalizer:
    """Zero-mean, unit-variance normalizer computed from training data."""

    def __init__(self, x_mean, x_std, y_mean, y_std):
        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_std  = np.asarray(x_std,  dtype=np.float32)
        self.y_mean = np.asarray(y_mean, dtype=np.float32)
        self.y_std  = np.asarray(y_std,  dtype=np.float32)

    def normalize_x(self, x):
        return (x - self.x_mean) / self.x_std

    def normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    def save(self, path: str):
        data = {
            "x_mean": self.x_mean.tolist(),
            "x_std":  self.x_std.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_std":  self.y_std.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"Normalizer saved: {path}")

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        data = json.loads(Path(path).read_text())
        return cls(data["x_mean"], data["x_std"], data["y_mean"], data["y_std"])


class KinematicsDataset:
    """Loads the UR3e kinematics HDF5 dataset and serves JAX-ready batches.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    x_field : str
        Dataset key for inputs  (default: 'x'  → shape [N, 18]).
    y_field : str
        Dataset key for outputs (default: 'y'  → shape [N, 15]).
    normalize : bool
        If True, apply normalizer when iterating. Call fit_normalizer() first.
    dtype : numpy dtype
        Precision for loaded arrays (default: float32).
    """

    def __init__(self, path: str,
                 x_field: str = X_FIELD,
                 y_field: str = Y_FIELD,
                 normalize: bool = False,
                 dtype=np.float32):
        self.path = path
        self.x_field = x_field
        self.y_field = y_field
        self.normalize = normalize
        self.dtype = dtype
        self.normalizer: Optional[Normalizer] = None

        # Pre-cache split sizes (no full load)
        self._sizes = {}
        with h5py.File(path, 'r') as f:
            for split in SPLITS:
                if split in f and x_field in f[split]:
                    self._sizes[split] = int(f[split][x_field].shape[0])
            self.x_dim = int(f[list(self._sizes.keys())[0]][x_field].shape[1])
            self.y_dim = int(f[list(self._sizes.keys())[0]][y_field].shape[1])
            self.metadata = dict(f["metadata"].attrs)

        print(f"Dataset: {path}")
        print(f"  x: {self.x_dim}-dim  |  y: {self.y_dim}-dim")
        for s, n in self._sizes.items():
            print(f"  {s:20s}: {n:>8,} samples")

    # ── Loading ─────────────────────────────────────────────────────────────
    def _load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a full split into memory as numpy arrays."""
        with h5py.File(self.path, 'r') as f:
            x = f[split][self.x_field][:].astype(self.dtype)
            y = f[split][self.y_field][:].astype(self.dtype)
        return x, y

    def get_split(self, split: str) -> Tuple[jax.Array, jax.Array]:
        """Return an entire split as a (x, y) pair of JAX arrays."""
        x, y = self._load_split(split)
        if self.normalize and self.normalizer is not None:
            x = self.normalizer.normalize_x(x)
            y = self.normalizer.normalize_y(y)
        return jnp.array(x), jnp.array(y)

    def size(self, split: str) -> int:
        return self._sizes.get(split, 0)

    # ── Normalizer ──────────────────────────────────────────────────────────
    def fit_normalizer(self, split: str = "train") -> Normalizer:
        """Compute mean/std from the training split and store as self.normalizer."""
        print(f"Fitting normalizer on '{split}' split ({self._sizes[split]:,} samples)...")
        x, y = self._load_split(split)
        x_std = x.std(axis=0)
        y_std = y.std(axis=0)
        # Avoid division by zero for constant features
        x_std = np.where(x_std < 1e-8, 1.0, x_std)
        y_std = np.where(y_std < 1e-8, 1.0, y_std)
        self.normalizer = Normalizer(x.mean(axis=0), x_std, y.mean(axis=0), y_std)
        print(f"  x_mean range: [{self.normalizer.x_mean.min():.4f}, {self.normalizer.x_mean.max():.4f}]")
        print(f"  y_mean range: [{self.normalizer.y_mean.min():.4f}, {self.normalizer.y_mean.max():.4f}]")
        return self.normalizer

    def save_normalizer(self, path: str):
        assert self.normalizer is not None, "Call fit_normalizer() first"
        self.normalizer.save(path)

    def load_normalizer(self, path: str):
        self.normalizer = Normalizer.load(path)

    # ── Batch iterator ──────────────────────────────────────────────────────
    def batches(self, split: str, batch_size: int = 256,
                shuffle: bool = True, rng_seed: int = 0,
                drop_last: bool = False) -> Iterator[Tuple[jax.Array, jax.Array]]:
        """Yield (x, y) JAX array batches from a split.

        Parameters
        ----------
        split      : one of 'train', 'val', 'test_random', etc.
        batch_size : number of samples per batch
        shuffle    : whether to shuffle indices each epoch
        rng_seed   : numpy random seed for reproducibility
        drop_last  : if True, drop the final incomplete batch
        """
        x_np, y_np = self._load_split(split)
        if self.normalize and self.normalizer is not None:
            x_np = self.normalizer.normalize_x(x_np)
            y_np = self.normalizer.normalize_y(y_np)

        n = len(x_np)
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(rng_seed)
            rng.shuffle(idx)

        for start in range(0, n, batch_size):
            end = start + batch_size
            if drop_last and end > n:
                break
            batch_idx = idx[start:end]
            yield jnp.array(x_np[batch_idx]), jnp.array(y_np[batch_idx])

    def epoch_batches(self, split: str, batch_size: int = 256,
                      n_epochs: int = 1, base_seed: int = 0,
                      drop_last: bool = True) -> Iterator[Tuple[int, jax.Array, jax.Array]]:
        """Yield (epoch, x, y) across multiple epochs with different shuffles."""
        for epoch in range(n_epochs):
            for x, y in self.batches(split, batch_size,
                                     shuffle=True, rng_seed=base_seed + epoch,
                                     drop_last=drop_last):
                yield epoch, x, y

    # ── Extra fields ────────────────────────────────────────────────────────
    def get_field(self, split: str, field: str) -> jax.Array:
        """Load any field from a split (e.g. 'ee_pos_world', 'q', 'J_world')."""
        with h5py.File(self.path, 'r') as f:
            arr = f[split][field][:].astype(self.dtype)
        return jnp.array(arr)

    def __repr__(self):
        return (f"KinematicsDataset('{self.path}', "
                f"x={self.x_dim}, y={self.y_dim}, "
                f"splits={list(self._sizes.keys())})")
