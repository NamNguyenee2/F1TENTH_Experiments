"""Causual MIMO DeepONet baseline.
https://ieeexplore.ieee.org/abstract/document/10394294
"""

import os
import sys
from pathlib import Path
from functools import partial
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import optax

DATA_NAME = "AC_PP_50000_07271410"
PHI_INDEX = 2
P = 20  
ALPHA = 20
HIDDEN_CELLS = 48
HORIZON = 20
SPLIT_SEED = 1
EPOCHS = 20000
LR = 5e-3
DECAY_RATE = 0.96
TRANSITION_STEPS = 200

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.auto_car.auto_car import f_phys


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
    horizon=5,
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
        x_lap   = x_data[ini:lap_end]
        u_lap   = u_data[ini:lap_end]
        ini     = lap_end

        for i in range(0, len(u_lap) - horizon):
            time_list.append(np.linspace(1, horizon, horizon, dtype=np.float32))
            initial_list.append(x_lap[i])
            input_list.append(u_lap[i : i + horizon])
            output_list.append(x_lap[i + 1 : i + horizon + 1])

    inputs  = np.asarray(input_list,   dtype=np.float32)
    initial = np.asarray(initial_list, dtype=np.float32)
    time    = np.asarray(time_list,    dtype=np.float32)
    output  = np.asarray(output_list,  dtype=np.float32)
    
    inputs = inputs.reshape(inputs.shape[0], -1)

    return {
        "inputs_train": jnp.asarray(inputs, dtype=jnp.float32),
        "initial_train":jnp.asarray(initial, dtype=jnp.float32),
        "time_train":   jnp.asarray(time, dtype=jnp.float32),
        "output_train": jnp.asarray(output, dtype=jnp.float32),
        "n_x": n_x,
        "n_u": n_u,
        "split_seed":   split_seed,
    }


def init_mimo_deeponet_params(
    key,
    branch_dims,
    state_dim,
    p = 2,
    output_dim = None,
    root_input_dim = None,
    branch_hidden = 128,
    trunk_hidden  = 128,
    root_hidden   = 128,
    constant_init=False,
):
    if not branch_dims:
        raise ValueError("At least one branch input dimension is required.")
    if len(set(branch_dims)) != 1:
        raise ValueError(
            "Packed branch evaluation requires every branch to have the same "
            f"input dimension; got {branch_dims}."
        )

    output_dim = state_dim if output_dim is None else output_dim
    root_input_dim = state_dim + 1 if root_input_dim is None else root_input_dim
    out_width  = p * output_dim
    branch_layer_count = 5
    keys = jax.random.split(key, len(branch_dims) * branch_layer_count + 6)

    offset = 0
    branches = []
    for branch_dim in branch_dims:
        branches.append(
            (
                init_dense(keys[offset], branch_dim, branch_hidden, constant_init),
                init_dense(keys[offset + 1], branch_hidden, branch_hidden, constant_init),
                init_dense(keys[offset + 2], branch_hidden, branch_hidden,  constant_init),
                init_dense(keys[offset + 3], branch_hidden, branch_hidden,  constant_init),
                init_dense(keys[offset + 4], branch_hidden, out_width,  constant_init),
            )
        )
        offset += branch_layer_count

    return {
        "branches": pack_branch_params(tuple(branches)),
        "trunk": (
            init_dense(keys[offset], 1,  trunk_hidden, constant_init),
            init_dense(keys[offset + 1], trunk_hidden, trunk_hidden, constant_init),
            init_dense(keys[offset + 2], trunk_hidden, out_width, constant_init),
        ),
        "root": (
            init_dense(keys[offset + 3], root_input_dim, root_hidden, constant_init),
            init_dense(keys[offset + 4], root_hidden, root_hidden, constant_init),
            init_dense(keys[offset + 5], root_hidden, out_width, constant_init),
        ),
    }


def pack_branch_params(branches):
    """Stack identically shaped branch MLPs for one batched evaluation."""
    if not branches:
        raise ValueError("At least one branch network is required.")

    layer_count = len(branches[0])
    if layer_count != 5 or any(len(branch) != layer_count for branch in branches):
        raise ValueError(
            "Nghiem branch networks must each contain exactly five layers."
        )

    try:
        return tuple(
            {
                "w": jnp.stack(tuple(branch[layer_idx]["w"] for branch in branches)),
                "b": jnp.stack(tuple(branch[layer_idx]["b"] for branch in branches)),
            }
            for layer_idx in range(layer_count)
        )
    except ValueError as exc:
        raise ValueError(
            "All Nghiem branch networks must have identical layer shapes for "
            "packed evaluation."
        ) from exc


def unpack_branch_params(packed_branches):
    """Return per-branch MLP parameters from the packed representation."""
    if len(packed_branches) != 5:
        raise ValueError("Packed Nghiem branches must contain exactly five layers.")
    branch_count = int(packed_branches[0]["w"].shape[0])
    return tuple(
        tuple(
            {"w": layer["w"][branch_idx], "b": layer["b"][branch_idx]}
            for layer in packed_branches
        )
        for branch_idx in range(branch_count)
    )


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


def dense(params, x):
    return x @ params["w"] + params["b"]


def root_initial_features(x_initial, root_input_dim=None):
    state_dim = x_initial.shape[-1]
    root_input_dim = state_dim + 1 if root_input_dim is None else root_input_dim
    if root_input_dim == state_dim:
        return x_initial
    if root_input_dim != state_dim + 1:
        raise ValueError(
            f"Unsupported root input dimension {root_input_dim} for state dimension {state_dim}"
        )

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


def causal_sigmoid_gate(t, horizon, alpha=10.):
    """Return sigma_i(t), which smoothly activates branch i at its sample time."""
    branch_idx = jnp.arange(1, horizon + 1, dtype=jnp.float32).reshape(1, -1)
    return jax.nn.sigmoid(alpha * (t - branch_idx + 0.5))


@partial(jax.jit, static_argnames=("branch_ranges", "p", "n_y"))
def predict_mimo_causal_init(params, x_inputs, x_time, x_initial, branch_ranges, p, n_y, alpha=10.):
    horizon = len(branch_ranges)
    branch_inputs = jnp.stack(
        tuple(x_inputs[:, start_idx:end_idx] for start_idx, end_idx in branch_ranges),
        axis=1,
    )
    branch_stack = jax.vmap(mlp5h, in_axes=(0, 1), out_axes=1)(
        params["branches"], branch_inputs
    ).reshape(-1, horizon, p, n_y)

    root_input = root_initial_features(x_initial, params["root"][0]["w"].shape[0])
    root_mat = mlp3h(params["root"], root_input).reshape(-1, p, n_y)
    time_flat = x_time.reshape(-1, 1)
    sigmoid_stack = causal_sigmoid_gate(time_flat, horizon, alpha).reshape(
        -1, horizon, horizon, 1, 1
    )

    # At each query time: sum_i sigma_i(t) * branch_net_i(u(t_i)).
    branch_star = jnp.sum(
        branch_stack[:, None, :, :, :] * sigmoid_stack,
        axis=2,
    )
    trunk_stack = mlp3h(params["trunk"], time_flat).reshape(
        -1, horizon, p, n_y
    )

    return jnp.sum(branch_star * trunk_stack * root_mat[:, None, :, :], axis=2) + x_initial[:, None, :]


@partial(jax.jit, static_argnames=("branch_ranges", "p", "n_y", "piml_weight", "alpha"))
def mse_loss(params, x_inputs, x_time, x_initial, y_true, branch_ranges, p, n_y, piml_weight=None, alpha=10.):
    y_pred = predict_mimo_causal_init(params, x_inputs, x_time, x_initial, branch_ranges, p, n_y, alpha)
    return jnp.mean((y_pred - y_true) ** 2)


@partial(jax.jit, static_argnames=("branch_ranges", "p", "n_y", "piml_weight", "alpha"))
def piml_mse_loss(params, x_inputs, x_time, x_initial, y_true, branch_ranges, p, n_y, piml_weight, alpha=10.):
    y_pred_model = predict_mimo_causal_init(params, x_inputs, x_time, x_initial, branch_ranges, p, n_y, alpha)
    horizon    = y_true.shape[1]
    n_u         = x_inputs.shape[1] // horizon
    u_seq       = x_inputs.reshape(x_inputs.shape[0], horizon, n_u)
    x_prev      = jnp.concatenate((x_initial[:, None, :], y_pred_model[:, :-1, :]), axis=1)
    y_pred_phys = jax.vmap(jax.vmap(f_phys))(x_prev, u_seq)

    data_loss    = jnp.mean((y_pred_model - y_true) ** 2)
    physics_loss = jnp.mean((y_pred_model[:,:,:2] - y_pred_phys[:,:,:2]) ** 2)
    return data_loss + piml_weight * physics_loss


def make_adam_optimizer(learning_rate, l2_reg=0.0):
    if l2_reg > 0.0:
        return optax.chain(optax.add_decayed_weights(l2_reg), optax.adam(learning_rate))
    return optax.adam(learning_rate)


def make_mimo_train_step_init(optimizer, branch_ranges, p, n_y, loss_fn=mse_loss, piml_weight = 0.1, alpha=10.):
    @jax.jit
    def train_step(params, opt_state, x_inputs, x_time, x_initial, y_true):
        loss_value, grads = jax.value_and_grad(loss_fn)(
            params, x_inputs, x_time, x_initial, y_true, branch_ranges, p, n_y, piml_weight, alpha
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    return train_step


def train_mimo_model_init(
    params,
    data,
    branch_ranges,
    p,
    n_y,
    lr=1e-2,
    epochs=1000,
    decay_rate = 0.95,
    transition_steps = 200,
    l2_reg = 1e-6,
    piml_weight = None,
    alpha = 10.,
):
    
    schedule = optax.exponential_decay(init_value=lr, transition_steps=transition_steps, decay_rate=decay_rate, staircase=True )

    optimizer = make_adam_optimizer(schedule, l2_reg)

    opt_state = optimizer.init(params)

    loss_fn = mse_loss if piml_weight is None else piml_mse_loss

    train_step = make_mimo_train_step_init(
        optimizer,
        branch_ranges,
        p,
        n_y,
        loss_fn,
        piml_weight=piml_weight,
        alpha=alpha,
    )
    
    losses = []

    for epoch in range(EPOCHS):
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
            print(f"Epoch: {epoch}/{EPOCHS} with RMSE loss: {float(np.sqrt(loss_value))}")

    return params, losses

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


def evaluate_mimo_model_init(params, data, branch_ranges, p, n_y, alpha=10.):

    data_len = data["inputs_train"].shape[0]
    
    test_len = 1000
    start = np.random.randint(1, int(data_len)-test_len)

    y_pred = predict_mimo_causal_init(
        params,
        data["inputs_train"][start:start+test_len],
        data["time_train"][start:start+test_len],
        data["initial_train"][start:start+test_len],
        branch_ranges,
        p,
        n_y,
        alpha,
        )
    y_true = data["output_train"][start:start+test_len]

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    channel_results = []
    for ch in range(n_y):
        results = []
        for posi in range(y_pred.shape[1]):
            pred   = y_pred[:, posi, ch]
            target = y_true[:, posi, ch]
            mse    = np.mean((pred - target) ** 2)
            mae    = max(np.abs(pred - target))
            results.append((posi, np.sqrt(mse), mae))
        channel_results.append((ch, results))

    return channel_results

def print_mimo_results(title, channel_results):
    print(title)
    for ch, results in channel_results:
        print(f"\nChannel {ch}")
        print(f"{'Position':<10}{'RMSE':<10}{'MAE':<10}")
        for posi, rmse, mma in results:
            print(f"{posi+1:<10}{rmse:<10.4f}{mma:<10.4f}")


@dataclass(frozen=True)
class DeepONetBundle:
    params: dict
    branch_ranges: tuple[tuple[int, int], ...]
    p: int
    n_x: int
    n_u: int
    alpha: float
    horizon: int
    root_input_dim: int | None = None


def _layer_key(prefix: str, index: int, name: str) -> str:
    return f"{prefix}_{index}_{name}"


def save_bundle(bundle: DeepONetBundle, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    branches = unpack_branch_params(bundle.params["branches"])
    root_input_dim = bundle.root_input_dim
    if root_input_dim is None:
        root_input_dim = int(np.asarray(bundle.params["root"][0]["w"]).shape[0])
    arrays: dict[str, np.ndarray] = {
        "branch_ranges": np.asarray(bundle.branch_ranges, dtype=np.int32),
        "p": np.asarray(bundle.p,               dtype=np.int32),
        "n_x": np.asarray(bundle.n_x,           dtype=np.int32),
        "n_u": np.asarray(bundle.n_u,           dtype=np.int32),
        "alpha": np.asarray(bundle.alpha,       dtype=np.float32),
        "horizon": np.asarray(bundle.horizon,   dtype=np.int32),
        "root_input_dim": np.asarray(root_input_dim, dtype=np.int32),
        "n_branches": np.asarray(len(branches),                             dtype=np.int32),
        "n_branch_layers": np.asarray(len(branches[0]),                     dtype=np.int32),
        "n_trunk_layers": np.asarray(len(bundle.params["trunk"]),           dtype=np.int32),
        "n_root_layers": np.asarray(len(bundle.params["root"]),             dtype=np.int32),
    }
    for b_idx, branch in enumerate(branches):
        for l_idx, layer in enumerate(branch):
            arrays[_layer_key(f"branch_{b_idx}", l_idx, "w")] = np.asarray(layer["w"], dtype=np.float32)
            arrays[_layer_key(f"branch_{b_idx}", l_idx, "b")] = np.asarray(layer["b"], dtype=np.float32)
    for group_name in ("trunk", "root"):
        for l_idx, layer in enumerate(bundle.params[group_name]):
            arrays[_layer_key(group_name, l_idx, "w")] = np.asarray(layer["w"], dtype=np.float32)
            arrays[_layer_key(group_name, l_idx, "b")] = np.asarray(layer["b"], dtype=np.float32)
    np.savez(path, **arrays)
    return str(path)


def load_bundle(path: str | Path) -> DeepONetBundle:
    with np.load(path) as data:
        n_branches = int(data["n_branches"])
        n_branch_layers = int(data["n_branch_layers"])
        if n_branch_layers != 5:
            raise ValueError(
                "This Nghiem implementation requires five-layer branch networks, "
                f"but {path} contains {n_branch_layers}. Retrain or supply a "
                "compatible checkpoint."
            )
        branches = []
        for b_idx in range(n_branches):
            layers = []
            for l_idx in range(n_branch_layers):
                layers.append({
                    "w": jnp.asarray(data[_layer_key(f"branch_{b_idx}", l_idx, "w")]),
                    "b": jnp.asarray(data[_layer_key(f"branch_{b_idx}", l_idx, "b")]),
                })
            branches.append(tuple(layers))

        expected_branch_width = int(data["p"]) * int(data["n_x"])
        for branch_idx, branch in enumerate(branches):
            actual_width = int(branch[-1]["w"].shape[-1])
            if actual_width != expected_branch_width:
                raise ValueError(
                    f"Branch {branch_idx} in {path} outputs {actual_width} values; "
                    f"expected p * n_x = {expected_branch_width}. Retrain or "
                    "supply a compatible checkpoint."
                )

        params = {"branches": pack_branch_params(tuple(branches))}
        for group_name in ("trunk", "root"):
            n_layers = int(data[f"n_{group_name}_layers"])
            params[group_name] = tuple(
                {
                    "w": jnp.asarray(data[_layer_key(group_name, l_idx, "w")]),
                    "b": jnp.asarray(data[_layer_key(group_name, l_idx, "b")]),
                }
                for l_idx in range(n_layers)
            )

        bundle = DeepONetBundle(
            params=params,
            branch_ranges=tuple(map(tuple, np.asarray(data["branch_ranges"], dtype=np.int32).tolist())),
            p=int(data["p"]),
            n_x=int(data["n_x"]),
            n_u=int(data["n_u"]),
            alpha=float(data["alpha"]),
            horizon=int(data["horizon"]),
            root_input_dim=(
                int(data["root_input_dim"])
                if "root_input_dim" in data.files
                else int(np.asarray(params["root"][0]["w"]).shape[0])
            ),
        )
        expected_ranges = tuple(
            (i * bundle.n_u, (i + 1) * bundle.n_u)
            for i in range(bundle.horizon)
        )
        if bundle.branch_ranges != expected_ranges:
            raise ValueError(
                "Each Nghiem baseline branch must receive exactly one u(t_i); "
                f"expected ranges {expected_ranges}, got {bundle.branch_ranges}."
            )
        branch_count = int(params["branches"][0]["w"].shape[0])
        if branch_count != bundle.horizon:
            raise ValueError(
                f"Expected {bundle.horizon} branch networks, got "
                f"{branch_count}."
            )
        return bundle


def predict_states(bundle: DeepONetBundle, x0: jnp.ndarray, u_seq: jnp.ndarray) -> jnp.ndarray:
    x0       = jnp.asarray(x0,    dtype=jnp.float32)
    u_seq    = jnp.asarray(u_seq, dtype=jnp.float32)
    x_inputs = u_seq.reshape(1, -1)
    x_time   = jnp.arange(1, bundle.horizon + 1, dtype=jnp.float32).reshape(1, -1)
    return predict_mimo_causal_init(
        bundle.params,
        x_inputs,
        x_time,
        x0.reshape(1, -1),
        bundle.branch_ranges,
        bundle.p,
        bundle.n_x,
        bundle.alpha,
    )[0]


def default_model_path(str = "auto_car_learning/DeepONet_ARC_default.npz") -> Path:
    return REPO_ROOT / "model" / str


def main():
    print("JAX devices:", jax.devices())

    model_path = default_model_path(
        f"auto_car_learning/PIML_DeepONet_nghiem2023_{DATA_NAME}"
        f"_p={p}_h={HIDDEN_CELLS}_alpha={alpha:g}.npz"
    )
    data_file = os.path.join(REPO_ROOT, "data/training", DATA_NAME + ".npz")

    x_data, u_data, lap_data, _ = load_arc_data(data_file)
    n_x = x_data.shape[1]
    n_u = u_data.shape[1]
    root_input_dim = n_x + 1


    start = 10000
    end   = 20000


    data = make_mimo_supervised_sequences(
        x_data,
        u_data,
        lap_data,
        start,
        end,
        horizon = HORIZON,
        split_seed = SPLIT_SEED,
    )

    # Branch i receives only the control sample u(t_i).
    branch_ranges = tuple((n_u * j, n_u * (j + 1)) for j in range(horizon))
    branch_dims   = tuple(end_idx - start_idx for start_idx, end_idx in branch_ranges)

    print("input", data["inputs_train"].shape)
    print("time",  data["time_train"].shape)
    print("init",  data["initial_train"].shape)
    print("nx = ", n_x, "nu = ", n_u)
    print("root_input_dim =", root_input_dim)
    print("branch_dims =", branch_dims)
    print("data_file =", data_file)
    print("split_seed =", SPLIT_SEED)

    params = init_mimo_deeponet_params(
        jax.random.PRNGKey(SPLIT_SEED),
        branch_dims=branch_dims,
        state_dim=n_x,
        p=p,
        output_dim=n_x,
        root_input_dim=root_input_dim,
        branch_hidden = HIDDEN_CELLS,
        trunk_hidden  = HIDDEN_CELLS,
        root_hidden   = HIDDEN_CELLS,
    )

    params, _ = train_mimo_model_init(
        params,
        data,
        branch_ranges = branch_ranges,
        p = P,
        n_y = n_x,
        lr = LR,
        epochs = EPOCHS,
        decay_rate = DECAY_RATE,
        transition_steps = TRANSITION_STEPS,
        piml_weight = 1.,
        alpha = ALPHA,
    )

    bundle = DeepONetBundle(
        params=params,
        branch_ranges=branch_ranges,
        p       = P,
        n_x     = n_x,
        n_u     = n_u,
        alpha   = ALPHA,
        horizon = HORIZON,
        root_input_dim = root_input_dim,
    )
    saved_path = save_bundle(bundle, model_path)
    print(f"Saved DeepONet v3 model to {saved_path}")

    bundle  = load_bundle(model_path)
    results = evaluate_mimo_model_init(
        bundle.params,
        data,
        bundle.branch_ranges,
        bundle.p,
        bundle.n_x,
        bundle.alpha,
    )
    print_mimo_results("Nghiem 2023 Causal MIMO DeepONet", results)



if __name__ == "__main__":
    main()
