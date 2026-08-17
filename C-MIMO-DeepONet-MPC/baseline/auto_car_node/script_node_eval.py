"""Evaluate a trained residual AutoCar NODE over H-step rollouts."""

from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.auto_car_deeponet.script_jong2025 import (  # noqa: E402
    load_arc_data,
    make_mimo_supervised_sequences,
)
from baseline.auto_car_node.script_node import (  # noqa: E402
    load_bundle,
    predict_states_batch,
)


DATA_NAME = "AC_PP_50000_07271410"
HORIZON = 20
HIDDEN_DIM = 96
MAX_SAMPLES = 1000
EVALUATION_SEED = 7


def evaluate_node(
    params,
    x_initial,
    u_seq,
    y_true,
    *,
    max_samples=MAX_SAMPLES,
    evaluation_seed=EVALUATION_SEED,
):
    """Report H-step errors for randomly selected trajectory windows."""
    total_trajectories = int(x_initial.shape[0])
    count = min(max_samples, total_trajectories)
    if count == 0:
        raise ValueError("Evaluation requires at least one trajectory.")

    sample_indices = np.random.default_rng(evaluation_seed).choice(
        total_trajectories, size=count, replace=False
    )
    x_eval = jnp.asarray(x_initial[sample_indices], dtype=jnp.float32)
    u_eval = jnp.asarray(u_seq[sample_indices], dtype=jnp.float32)
    prediction = np.asarray(predict_states_batch(params, x_eval, u_eval))
    target = np.asarray(y_true[sample_indices])

    print("Residual NODE H-step evaluation")
    print(
        f"Randomly selected {count} of {total_trajectories} H-step "
        f"trajectories with seed {evaluation_seed}"
    )
    channel_names = ("X", "Y", "phi", "v")
    for channel, name in enumerate(channel_names):
        print(f"\nChannel {channel} ({name})")
        print(f"{'Position':<10}{'RMSE':<12}{'MaxAE':<12}")
        for step in range(target.shape[1]):
            error = prediction[:, step, channel] - target[:, step, channel]
            print(
                f"{step + 1:<10}{np.sqrt(np.mean(error ** 2)):<12.4f}"
                f"{np.max(np.abs(error)):<12.4f}"
            )


def main():
    model_path = (
        REPO_ROOT
        / "model"
        / "auto_car_learning"
        / f"PIML_NODE_{DATA_NAME}_h={HORIZON}_hidden={HIDDEN_DIM}.npz"
    )
    bundle = load_bundle(model_path)
    print(f"Loaded PIML Neural ODE model from {model_path}")

    data_file = REPO_ROOT / "data" / "training" / f"{DATA_NAME}.npz"
    x_data, u_data, lap_data = load_arc_data(data_file)
    data = make_mimo_supervised_sequences(
        x_data,
        u_data,
        lap_data,
        10_000,
        20_000,
        num_pred=bundle.horizon,
        split_seed=EVALUATION_SEED,
    )
    u_seq = data["inputs_train"].reshape(
        -1, bundle.horizon, bundle.n_u
    )

    evaluate_node(
        bundle.params,
        data["initial_train"],
        u_seq,
        data["output_train"],
    )


if __name__ == "__main__":
    main()
