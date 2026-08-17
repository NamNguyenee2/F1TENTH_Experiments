from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import partial
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import optax


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.auto_car.auto_car import f_phys

PHI_INDEX = 2

@dataclass(frozen=True)
class DeepONetBundle:
    params: dict
    p: int
    n_x: int
    n_u: int
    horizon: int


def _layer_key(prefix: str, index: int, name: str) -> str:
    return f"{prefix}_{index}_{name}"


def load_arc_data(name_str):
    data_path = os.path.join(REPO_ROOT, "data/training", name_str+".npz")
    with np.load(data_path) as archive:
        print(archive.files)
        x_data = archive["x"].astype(np.float32)
        u_data = archive["u"].astype(np.float32)
        lap_data = archive["lap_sample_counts"].astype(np.int32)
        sample_time = archive["sample_time"].astype(np.float32)

    return x_data, u_data, lap_data, sample_time


def make_mimo_supervised_sequences(
    x_data,
    u_data,
    lap_data,
    start,
    end,
    num_pred=5,
    split_seed=0,
):
    x_data = np.asarray(x_data, dtype=np.float32)
    u_data = np.asarray(u_data, dtype=np.float32)
    n_x = x_data.shape[1]
    n_u = u_data.shape[1]

    input_list = []
    output_list = []
    initial_list = []
    time_list = []

    ini = 0
    for count in lap_data:
        lap_end = ini + int(count)
        x_lap = x_data[ini:lap_end]
        u_lap = u_data[ini:lap_end]
        ini = lap_end

        for i in range(0, len(u_lap) - num_pred):
            time_list.append(np.linspace(1, num_pred, num_pred, dtype=np.float32))
            initial_list.append(x_lap[i])
            input_list.append(u_lap[i : i + num_pred])
            output_list.append(x_lap[i + 1 : i + num_pred + 1])

    inputs = np.asarray(input_list, dtype=np.float32)
    initial = np.asarray(initial_list, dtype=np.float32)
    time = np.asarray(time_list, dtype=np.float32)
    output = np.asarray(output_list, dtype=np.float32)

    inputs = inputs.reshape(inputs.shape[0], -1)

    return {
        "inputs_train": jnp.asarray(inputs, dtype=jnp.float32),
        "initial_train": jnp.asarray(initial, dtype=jnp.float32),
        "time_train": jnp.asarray(time, dtype=jnp.float32),
        "output_train": jnp.asarray(output, dtype=jnp.float32),
        "n_x": n_x,
        "n_u": n_u,
        "split_seed": split_seed,
    }


def init_dense(key, in_dim, out_dim, constant_init=False):
    if constant_init:
        return {
            "w": jnp.full((in_dim, out_dim), 0.5, dtype=jnp.float32),
            "b": jnp.full((out_dim,), 0.5, dtype=jnp.float32),
        }

    w_key, b_key = jax.random.split(key)
    limit = 1.0 / jnp.sqrt(float(in_dim))
    return {
        "w": jax.random.uniform(
            w_key, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=jnp.float32
        ),
        "b": jax.random.uniform(
            b_key, (out_dim,), minval=-limit, maxval=limit, dtype=jnp.float32
        ),
    }


def init_mimo_deeponet_params(
    key,
    control_input_dim,
    state_dim,
    p=2,
    output_dim=None,
    branch_hidden=128,
    trunk_hidden=128,
    constant_init=False,
):
    output_dim = state_dim if output_dim is None else output_dim
    initial_feature_dim = state_dim + 1
    out_width = p * output_dim
    keys = jax.random.split(key, 8)

    return {
        "branch": (
            init_dense(
                keys[0],
                control_input_dim + initial_feature_dim,
                branch_hidden,
                constant_init,
            ),
            init_dense(keys[1], branch_hidden, branch_hidden, constant_init),
            init_dense(keys[2], branch_hidden, branch_hidden, constant_init),
            init_dense(keys[3], branch_hidden, branch_hidden, constant_init),
            init_dense(keys[4], branch_hidden, out_width, constant_init),
        ),
        "trunk": (
            init_dense(keys[5], 1, trunk_hidden, constant_init),
            init_dense(keys[6], trunk_hidden, trunk_hidden, constant_init),
            init_dense(keys[7], trunk_hidden, out_width, constant_init),
        ),
    }


def dense(params, x):
    return x @ params["w"] + params["b"]


def mlp3h(params, x):
    x = jax.nn.gelu(dense(params[0], x))
    x = jax.nn.silu(dense(params[1], x))
    return dense(params[2], x)


def mlp5h(params, x):
    x = jax.nn.gelu(dense(params[0], x))
    x = jax.nn.silu(dense(params[1], x))
    x = jax.nn.gelu(dense(params[2], x))
    x = jax.nn.silu(dense(params[3], x))
    return dense(params[4], x)


def initial_features(x_initial):
    phi = x_initial[..., PHI_INDEX : PHI_INDEX + 1]
    return jnp.concatenate(
        (
            x_initial[..., :PHI_INDEX],
            jnp.sin(phi),
            jnp.cos(phi),
            x_initial[..., PHI_INDEX + 1 :],
        ),
        axis=-1,
    )


@partial(jax.jit, static_argnames=("p", "n_y"))
def predict_mimo_unstack_init(
    params,
    x_inputs,
    x_time,
    x_initial,
    p,
    n_y,
):
    x0_features   = initial_features(x_initial)
    branch_inputs = jnp.concatenate((x_inputs, x0_features), axis=-1)
    branch_stack  = mlp5h(params["branch"], branch_inputs).reshape(
        x_inputs.shape[0], 1, p, n_y
    )

    time_flat = x_time.reshape(-1, 1)
    trunk_stack = mlp3h(params["trunk"], time_flat).reshape(
        x_time.shape[0], x_time.shape[1], p, n_y
    )

    return jnp.sum(branch_stack * trunk_stack, axis=2) + x_initial[:, None, :]


@partial(jax.jit, static_argnames=("p", "n_y", "piml_weight"))
def mse_loss(
    params,
    x_inputs,
    x_time,
    x_initial,
    y_true,
    p,
    n_y,
    piml_weight=None,
):
    y_pred = predict_mimo_unstack_init(params, x_inputs, x_time, x_initial, p, n_y)
    return jnp.mean((y_pred - y_true) ** 2)


@partial(jax.jit, static_argnames=("p", "n_y", "piml_weight"))
def piml_mse_loss(
    params,
    x_inputs,
    x_time,
    x_initial,
    y_true,
    p,
    n_y,
    piml_weight,
):
    y_pred_model = predict_mimo_unstack_init(
        params, x_inputs, x_time, x_initial, p, n_y
    )
    num_pred = y_true.shape[1]
    n_u = x_inputs.shape[1] // num_pred
    u_seq = x_inputs.reshape(x_inputs.shape[0], num_pred, n_u)
    x_prev = jnp.concatenate((x_initial[:, None, :], y_pred_model[:, :-1, :]), axis=1)
    y_pred_phys = jax.vmap(jax.vmap(f_phys))(x_prev, u_seq)

    data_loss = jnp.mean((y_pred_model - y_true) ** 2)
    physics_loss = jnp.mean((y_pred_model - y_pred_phys) ** 2)
    return data_loss + piml_weight * physics_loss


def make_adam_optimizer(learning_rate, l2_reg=0.0):
    if l2_reg > 0.0:
        return optax.chain(optax.add_decayed_weights(l2_reg), optax.adam(learning_rate))
    return optax.adam(learning_rate)


def make_mimo_train_step_init(
    optimizer,
    p,
    n_y,
    loss_fn=mse_loss,
    piml_weight=None,
):
    @jax.jit
    def train_step(params, opt_state, x_inputs, x_time, x_initial, y_true):
        loss_value, grads = jax.value_and_grad(loss_fn)(
            params,
            x_inputs,
            x_time,
            x_initial,
            y_true,
            p,
            n_y,
            piml_weight,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    return train_step


def train_mimo_model_init(
    params,
    data,
    p,
    n_y,
    lr=1e-2,
    epochs=1000,
    decay_rate=0.95,
    transition_steps=200,
    l2_reg=1e-6,
    piml_weight=None,
):
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        staircase=True,
    )

    optimizer = make_adam_optimizer(schedule, l2_reg)
    opt_state = optimizer.init(params)
    loss_fn = mse_loss if piml_weight is None else piml_mse_loss
    train_step = make_mimo_train_step_init(
        optimizer,
        p,
        n_y,
        loss_fn,
        piml_weight=piml_weight,
    )

    losses = []
    for epoch in range(epochs):
        params, opt_state, loss_value = train_step(
            params,
            opt_state,
            data["inputs_train"],
            data["time_train"],
            data["initial_train"],
            data["output_train"],
        )
        losses.append(float(loss_value))
        if epoch % transition_steps == 0:
            print(f"Epoch: {epoch}/{epochs} with RMSE loss: {float(np.sqrt(loss_value))}")

    return params, losses


def evaluate_mimo_model_init(params, data, p, n_y):
    data_len = data["inputs_train"].shape[0]
    test_len = min(1000, int(data_len) - 1)
    if test_len <= 0:
        raise ValueError("Need at least two supervised samples to evaluate the model.")
    start = np.random.randint(0, int(data_len) - test_len + 1)

    y_pred = predict_mimo_unstack_init(
        params,
        data["inputs_train"][start : start + test_len],
        data["time_train"][start : start + test_len],
        data["initial_train"][start : start + test_len],
        p,
        n_y,
    )
    y_true = data["output_train"][start : start + test_len]

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    channel_results = []
    for ch in range(n_y):
        results = []
        for posi in range(y_pred.shape[1]):
            pred = y_pred[:, posi, ch]
            target = y_true[:, posi, ch]
            mse = np.mean((pred - target) ** 2)
            mae = max(np.abs(pred - target))
            results.append((posi, np.sqrt(mse), mae))
        channel_results.append((ch, results))

    return channel_results


def print_mimo_results(title, channel_results):
    print(title)
    for ch, results in channel_results:
        print(f"\nChannel {ch}")
        print(f"{'Position':<10}{'RMSE':<10}{'MAE':<10}")
        for posi, rmse, mma in results:
            print(f"{posi + 1:<10}{rmse:<10.4f}{mma:<10.4f}")


def save_bundle(bundle: DeepONetBundle, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "p": np.asarray(bundle.p, dtype=np.int32),
        "n_x": np.asarray(bundle.n_x, dtype=np.int32),
        "n_u": np.asarray(bundle.n_u, dtype=np.int32),
        "horizon": np.asarray(bundle.horizon, dtype=np.int32),
        "n_branch_layers": np.asarray(len(bundle.params["branch"]), dtype=np.int32),
        "n_trunk_layers": np.asarray(len(bundle.params["trunk"]), dtype=np.int32),
        "n_root_layers": np.asarray(0, dtype=np.int32),
    }
    for group_name in ("branch", "trunk"):
        for layer_idx, layer in enumerate(bundle.params[group_name]):
            arrays[_layer_key(group_name, layer_idx, "w")] = np.asarray(
                layer["w"], dtype=np.float32
            )
            arrays[_layer_key(group_name, layer_idx, "b")] = np.asarray(
                layer["b"], dtype=np.float32
            )
    np.savez(path, **arrays)
    return str(path)


def load_bundle(path: str | Path) -> DeepONetBundle:
    with np.load(path) as data:
        n_root_layers = int(data["n_root_layers"]) if "n_root_layers" in data.files else 0
        if n_root_layers:
            raise ValueError("deeponet_unstack_baseline.load_bundle expected a rootless bundle")

        params = {}
        n_u = int(data["n_u"])
        for group_name in ("branch", "trunk"):
            n_layers = int(data[f"n_{group_name}_layers"])
            params[group_name] = tuple(
                {
                    "w": jnp.asarray(data[_layer_key(group_name, layer_idx, "w")]),
                    "b": jnp.asarray(data[_layer_key(group_name, layer_idx, "b")]),
                }
                for layer_idx in range(n_layers)
            )

        bundle = DeepONetBundle(
            params=params,
            p=int(data["p"]),
            n_x=int(data["n_x"]),
            n_u=n_u,
            horizon=int(data["horizon"]),
        )
        expected_branch_width = bundle.horizon * bundle.n_u + bundle.n_x + 1
        actual_branch_width = int(params["branch"][0]["w"].shape[0])
        if actual_branch_width != expected_branch_width:
            raise ValueError(
                "Unstack baseline branch input must contain the complete control "
                f"horizon and initial-state features (width {expected_branch_width}), "
                f"but found width {actual_branch_width}."
            )
        return bundle


def predict_states(bundle: DeepONetBundle, x0: jnp.ndarray, u_seq: jnp.ndarray) -> jnp.ndarray:
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    u_seq = jnp.asarray(u_seq, dtype=jnp.float32)
    x_inputs = u_seq.reshape(1, -1)
    x_time = jnp.arange(1, bundle.horizon + 1, dtype=jnp.float32).reshape(1, -1)
    return predict_mimo_unstack_init(
        bundle.params,
        x_inputs,
        x_time,
        x0.reshape(1, -1),
        bundle.p,
        bundle.n_x,
    )[0]


def predict_states_u_jacobian(
    bundle: DeepONetBundle, x0: jnp.ndarray, u_seq: jnp.ndarray
) -> jnp.ndarray:
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    u_seq = jnp.asarray(u_seq, dtype=jnp.float32)

    def predict_from_u(u_flat):
        return predict_states(
            bundle,
            x0,
            u_flat.reshape((bundle.horizon, bundle.n_u)),
        ).reshape(-1)

    jac = jax.jacfwd(predict_from_u)(u_seq.reshape(-1))
    return jac.reshape(bundle.horizon, bundle.n_x, bundle.horizon, bundle.n_u)


def default_model_path(str="auto_car_learning/DeepONet_ARC_unstack_baseline_default.npz") -> Path:
    return REPO_ROOT / "model" / str


def main():
    print("JAX devices:", jax.devices())

    data_name = "AC_PP_50000_07271410"

    p   = 20
    hidden_ncells = 96
    branch_hidden = hidden_ncells
    trunk_hidden  = hidden_ncells

    num_pred    = 20
    split_seed  = 7
    epochs      = 20000
    lr          = 5e-3
    decay_rate  = 0.96
    transition_steps = 200
    l2_reg      = 1e-6
    piml_weight = 0.1

    data_file = os.path.join(REPO_ROOT, "data/training", data_name + ".npz")

    x_data, u_data, lap_data, _ = load_arc_data(data_file)
    n_x = x_data.shape[1]
    n_u = u_data.shape[1]
    model_path = default_model_path(
        "auto_car_learning/"
        f"PIML_DeepONet_jong2025_{data_name}_p={p}"
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

    initial_feature_dim = n_x + 1
    control_input_dim = num_pred * n_u
    branch_input_dim = control_input_dim + initial_feature_dim

    print("input", data["inputs_train"].shape)
    print("time", data["time_train"].shape)
    print("init", data["initial_train"].shape)
    print("nx = ", n_x, "nu = ", n_u)
    print("initial_feature_dim =", initial_feature_dim)
    print("control_input_dim =", control_input_dim)
    print("branch_input_dim =", branch_input_dim)
    print("branch_hidden =", branch_hidden)
    print("trunk_hidden =", trunk_hidden)
    print("data_file =", data_file)
    print("split_seed =", split_seed)

    params = init_mimo_deeponet_params(
        jax.random.PRNGKey(split_seed),
        control_input_dim=control_input_dim,
        state_dim=n_x,
        p=p,
        output_dim=n_x,
        branch_hidden=branch_hidden,
        trunk_hidden=trunk_hidden,
    )

    params, _ = train_mimo_model_init(
        params,
        data,
        p=p,
        n_y=n_x,
        lr=lr,
        epochs=epochs,
        decay_rate=decay_rate,
        transition_steps=transition_steps,
        l2_reg=l2_reg,
        piml_weight=piml_weight,
    )

    bundle = DeepONetBundle(
        params=params,
        p=p,
        n_x=n_x,
        n_u=n_u,
        horizon=num_pred,
    )
    saved_path = save_bundle(bundle, model_path)
    print(f"Saved standard unstack DeepONet baseline model to {saved_path}")

    bundle = load_bundle(model_path)
    results = evaluate_mimo_model_init(
        bundle.params,
        data,
        bundle.p,
        bundle.n_x,
    )
    print_mimo_results("Standard Unstack MIMO DeepONet Baseline", results)


if __name__ == "__main__":
    main()
