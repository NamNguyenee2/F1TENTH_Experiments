import os
import sys
from pathlib import Path
import yaml
import jax


DATA_NAME = "AC_PP_50000_08151455"
P = 20
TAU = 1
ENCODE_HIDDEN = 32
BRANCH_HIDDEN = 96
TRUNK_HIDDEN = 96
SEED = 10


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPCC_CONFIG = REPO_ROOT / "model" / "auto_car" / "mpcc_parameter.yaml"

with open(MPCC_CONFIG, "r") as file:
    param = yaml.safe_load(file)

HORIZON = int(param["H"])

from src.deeponet import (
    DeepONetBundle,
    default_model_path,
    evaluate_mimo_model_init,
    init_mimo_deeponet_params,
    load_arc_data,
    load_bundle,
    make_mimo_supervised_sequences,
    print_mimo_results,
    save_bundle,
    train_mimo_model_init,
)


def main():
    print("JAX devices:", jax.devices())

    data_name = DATA_NAME
    p   = P
    tau = TAU
    encode_hidden = ENCODE_HIDDEN
    branch_hidden = BRANCH_HIDDEN
    trunk_hidden  = TRUNK_HIDDEN

    num_pred    = HORIZON
    split_seed  = SEED
    epochs      = 20000
    lr          = 5e-3
    decay_rate  = 0.96
    transition_steps = 200
    l2_reg      = 1e-6
    piml_weight = 1.0

    data_file = os.path.join(REPO_ROOT, "data/training", data_name+".npz")

    x_data, u_data, lap_data, sample_time = load_arc_data(data_file)
    n_x = x_data.shape[1]
    n_u = u_data.shape[1]
    encode_output_dim = n_u
    model_path = default_model_path(
        "auto_car_learning/"
        f"CATR_MIMO_DeepONet_{data_name}_p={p}"
        f"_b={tau:g}_e={encode_hidden}"
        f"_br={branch_hidden}_tr={trunk_hidden}.npz"
    )

    start = 10000
    end = 20000

    data = make_mimo_supervised_sequences(
        x_data,
        u_data,
        lap_data,
        start,
        end,
        num_pred=num_pred,
        split_seed=split_seed,
    )

    branch_ranges = tuple((n_u * j, n_u * (j + 1)) for j in range(num_pred))
    initial_feature_dim = n_x + 1
    encode_input_dims = tuple(
        end_idx - start_idx
        for start_idx, end_idx in branch_ranges
    )
    branch_input_dim = num_pred * encode_output_dim + initial_feature_dim

    print("input", data["inputs_train"].shape)
    print("time", data["time_train"].shape)
    print("init", data["initial_train"].shape)
    print("nx = ", n_x, "nu = ", n_u)
    print("initial_feature_dim =", initial_feature_dim)
    print("encode_input_dims =", encode_input_dims)
    print("shared_encoder_input_dim =", n_u + 1)
    print("branch_input_dim =", branch_input_dim)
    print("encode_hidden =", encode_hidden)
    print("encode_output_dim =", encode_output_dim)
    print("branch_hidden =", branch_hidden)
    print("trunk_hidden =", trunk_hidden)
    print("gate_b =", tau)
    print("sample_time =", sample_time)
    print("data_file =", data_file)
    print("split_seed =", split_seed)

    params = init_mimo_deeponet_params(
        jax.random.PRNGKey(split_seed),
        encode_input_dims=encode_input_dims,
        state_dim=n_x,
        p=p,
        output_dim=n_x,
        encode_hidden=encode_hidden,
        encode_output_dim=encode_output_dim,
        branch_hidden=branch_hidden,
        trunk_hidden=trunk_hidden,
    )

    params, _ = train_mimo_model_init(
        params,
        data,
        branch_ranges=branch_ranges,
        p=p,
        n_y=n_x,
        lr=lr,
        epochs=epochs,
        decay_rate=decay_rate,
        transition_steps=transition_steps,
        l2_reg=l2_reg,
        piml_weight=piml_weight,
        tau=tau,
        sample_time=sample_time,
    )

    bundle = DeepONetBundle(
        params=params,
        branch_ranges=branch_ranges,
        p=p,
        n_x=n_x,
        n_u=n_u,
        tau=tau,
        horizon=num_pred,
        sample_time=sample_time,
    )
    saved_path = save_bundle(bundle, model_path)
    print(f"Saved shared-encoder causal-operator DeepONet model to {saved_path}")

    bundle = load_bundle(model_path)
    results = evaluate_mimo_model_init(
        bundle.params,
        data,
        bundle.branch_ranges,
        bundle.p,
        bundle.n_x,
        tau=bundle.tau,
        sample_time=bundle.sample_time,
    )
    print_mimo_results("Shared-Encoder Causal-Operator MIMO DeepONet", results)


if __name__ == "__main__":
    main()
