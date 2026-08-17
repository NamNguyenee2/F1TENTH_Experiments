"""Dataset splitting utilities (currently handled inline in generator; reserved for future use)."""
import numpy as np


def shuffle_split(data: dict, rng: np.random.Generator, fracs=(0.8, 0.1, 0.1)):
    """Split a dict-of-arrays into (train, val, test) by row index."""
    n = len(next(iter(data.values())))
    idx = rng.permutation(n)
    n_train = int(n * fracs[0])
    n_val = int(n * fracs[1])
    splits = []
    for start, end in [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, n)]:
        sub = {k: v[idx[start:end]] for k, v in data.items()}
        splits.append(sub)
    return splits
