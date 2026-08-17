"""Causal-operator C-MIMO DeepONet with a shared control encoder."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
import sys
import os
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
    branch_ranges: tuple[tuple[int, int], ...]
    p: int
    n_x: int
    n_u: int
    tau: float
    horizon: int
    sample_time: float = 0.05

    @property
    def alpha(self) -> float:
        """Compatibility alias for code that stores the causal sharpness as alpha."""
        return self.tau


@dataclass(frozen=True)
class CausalFeatures:
    """Broadcastable time features reused by CATR-MIMO predictions."""

    gates: np.ndarray
    encoder_times: np.ndarray
    trunk_times: np.ndarray


def _layer_key(prefix: str, index: int, name: str) -> str:
    return f"{prefix}_{index}_{name}"


def load_arc_data(data_path):
    with np.load(data_path) as archive:
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
    encode_input_dims,
    state_dim,
    p=2,
    output_dim=None,
    encode_hidden=128,
    encode_output_dim=None,
    branch_hidden=128,
    trunk_hidden=128,
    constant_init=False,
):
    output_dim = state_dim if output_dim is None else output_dim
    if len(encode_input_dims) == 0:
        raise ValueError("encode_input_dims must contain at least one encoder input dimension.")
    if any(dim != encode_input_dims[0] for dim in encode_input_dims):
        raise ValueError("CauOp Encode_net inputs must all have the same dimension.")
    encode_output_dim = encode_input_dims[0] if encode_output_dim is None else encode_output_dim
    initial_feature_dim = state_dim + 1
    out_width = p * output_dim
    keys = jax.random.split(key, 10)

    # Every control in the horizon is transformed by this one shared encoder.
    # Keeping a single parameter subtree is important: duplicating identical
    # initial values would still let the encoders diverge during optimization.
    encoders = (
        (
            init_dense(keys[0], encode_input_dims[0] + 1, encode_hidden, constant_init),
            init_dense(keys[1], encode_hidden, encode_output_dim, constant_init),
        ),
    )
    offset = 2

    return {
        "encoders": tuple(encoders),
        "branch": (
            init_dense(
                keys[offset],
                len(encode_input_dims) * encode_output_dim + initial_feature_dim,
                branch_hidden,
                constant_init,
            ),
            init_dense(keys[offset + 1], branch_hidden, branch_hidden, constant_init),
            init_dense(keys[offset + 2], branch_hidden, branch_hidden, constant_init),
            init_dense(keys[offset + 3], branch_hidden, branch_hidden, constant_init),
            init_dense(keys[offset + 4], branch_hidden, out_width, constant_init),
        ),
        "trunk": (
            init_dense(keys[offset + 5], 1, trunk_hidden, constant_init),
            init_dense(keys[offset + 6], trunk_hidden, trunk_hidden, constant_init),
            init_dense(keys[offset + 7], trunk_hidden, out_width, constant_init),
        ),
    }


def dense(params, x):
    return x @ params["w"] + params["b"]


def mlp_encode(params, x):
    x = jax.nn.gelu(dense(params[0], x))
    return dense(params[1], x)


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


def smooth_one_sided_causal_gate(delta_t, tau=1.0):
    """Causal gate h_b(delta_t) with d=1: 0, smoothstep, then 1."""
    b = tau
    safe_b = jnp.maximum(b, jnp.finfo(jnp.float32).eps)
    s = delta_t / safe_b
    smooth = 3.0 * s**2 - 2.0 * s**3
    return jnp.where(delta_t <= 0.0, 0.0, jnp.where(delta_t >= safe_b, 1.0, smooth))


def precompute_causal_features(
    x_time,
    horizon,
    tau=1.0,
    sample_time=0.05,
    dtype=np.float32,
):
    """Precompute broadcastable gates and encoder timestamps on the host."""
    query_times = np.asarray(x_time, dtype=dtype)
    if query_times.ndim == 1:
        query_times = query_times[None, :]
    if query_times.ndim != 2 or query_times.shape[1] == 0:
        raise ValueError("x_time must have shape (batch, query_steps) or (query_steps,).")
    if query_times.shape[0] > 1:
        if not np.all(query_times == query_times[:1]):
            raise ValueError("Precomputed causal features require a shared query-time grid.")
        query_times = query_times[:1]
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    causal_sample_times = np.arange(horizon, dtype=dtype)
    delta_t = query_times[:, :, None] - causal_sample_times[None, None, :]
    scaled_delta = delta_t / np.asarray(tau, dtype=dtype)
    smooth = 3.0 * scaled_delta**2 - 2.0 * scaled_delta**3
    gates = np.where(
        delta_t <= 0.0,
        0.0,
        np.where(delta_t >= tau, 1.0, smooth),
    )[..., None]
    encoder_times = (
        np.arange(horizon, dtype=dtype)[None, :, None]
        * np.asarray(sample_time, dtype=dtype)
    )
    return CausalFeatures(
        # Keep cached features as host arrays.  Converting them to JAX arrays
        # while this function is first called from a jitted computation would
        # cache DynamicJaxprTracers and leak them out of that trace.
        gates=np.asarray(gates, dtype=np.float32),
        encoder_times=np.asarray(encoder_times, dtype=np.float32),
        trunk_times=np.asarray(query_times[:, :, None], dtype=np.float32),
    )


@lru_cache(maxsize=16)
def standard_causal_features(horizon, tau=1.0, sample_time=0.05):
    """Return cached features for the standard query grid 1, ..., horizon."""
    query_times = np.arange(1, horizon + 1, dtype=np.float32)
    return precompute_causal_features(
        query_times,
        horizon,
        tau=tau,
        sample_time=sample_time,
    )


def causality_operator(encode_stack, x_time=None, tau=1.0, gates=None):
    """Map z_i to the block latent Z(t)=sum_i h_b(t-t_i) e_i z_i."""
    num_pred     = encode_stack.shape[1]
    if gates is None:
        if x_time is None:
            raise ValueError("x_time is required when causal gates are not precomputed.")
        sample_times = jnp.arange(num_pred, dtype=jnp.float32).reshape(1, -1)
        time_flat = x_time.reshape(-1, 1)
        gates = smooth_one_sided_causal_gate(time_flat - sample_times, tau)
        gates = gates.reshape(x_time.shape[0], x_time.shape[1], num_pred, 1)
    elif gates.shape[2] != num_pred:
        raise ValueError(
            f"Causal gates use horizon {gates.shape[2]}, expected {num_pred}."
        )
    z_blocks = encode_stack[:, None, :, :] * gates
    return z_blocks.reshape(encode_stack.shape[0], gates.shape[1], -1)


@partial(jax.jit, static_argnames=("branch_ranges", "p", "n_y"))
def predict_mimo_causal_init(
    params,
    x_inputs,
    x_time,
    x_initial,
    branch_ranges,
    p,
    n_y,
    tau=1.0,
    sample_time=0.05,
    causal_gates=None,
    encoder_time_features=None,
    trunk_time_features=None,
):
    x0_features = initial_features(x_initial)
    ranges = np.asarray(branch_ranges, dtype=np.int32)
    if ranges.shape != (len(branch_ranges), 2) or len(branch_ranges) == 0:
        raise ValueError("branch_ranges must contain at least one (start, end) pair.")
    branch_widths = ranges[:, 1] - ranges[:, 0]
    if np.any(branch_widths != branch_widths[0]):
        raise ValueError("The shared encoder requires equal-width branch ranges.")
    branch_width = int(ranges[0, 1] - ranges[0, 0])
    control_indices = jnp.asarray(
        ranges[:, :1] + np.arange(branch_width, dtype=np.int32)[None, :]
    )
    control_stack = jnp.take(x_inputs, control_indices, axis=1)
    if encoder_time_features is None:
        encoder_time_features = (
            jnp.arange(len(branch_ranges), dtype=x_inputs.dtype)[None, :, None]
            * sample_time
        )
    sample_times = jnp.broadcast_to(
        encoder_time_features,
        (x_inputs.shape[0], len(branch_ranges), 1),
    )
    encoder_inputs = jnp.concatenate((control_stack, sample_times), axis=-1)
    encode_stack = mlp_encode(params["encoders"][0], encoder_inputs)

    z_time = causality_operator(
        encode_stack,
        x_time=x_time,
        tau=tau,
        gates=causal_gates,
    )
    x0_time = jnp.broadcast_to(
        x0_features[:, None, :],
        (x0_features.shape[0], x_time.shape[1], x0_features.shape[-1]),
    )
    branch_inputs = jnp.concatenate((z_time, x0_time), axis=-1)
    branch_stack = mlp5h(params["branch"], branch_inputs.reshape(-1, branch_inputs.shape[-1]))
    branch_stack = branch_stack.reshape(x_time.shape[0], x_time.shape[1], p, n_y)

    if trunk_time_features is None:
        time_flat = x_time.reshape(-1, 1)
    else:
        time_flat = jnp.broadcast_to(
            trunk_time_features,
            (x_inputs.shape[0], x_time.shape[1], 1),
        ).reshape(-1, 1)
    trunk_stack = mlp3h(params["trunk"], time_flat).reshape(
        x_time.shape[0], x_time.shape[1], p, n_y
    )

    return jnp.sum(branch_stack * trunk_stack, axis=2) + x_initial[:, None, :]


@partial(jax.jit, static_argnames=("branch_ranges", "p", "n_y", "piml_weight", "tau"))
def mse_loss(
    params,
    x_inputs,
    x_time,
    x_initial,
    y_true,
    branch_ranges,
    p,
    n_y,
    piml_weight=None,
    tau=1.0,
    sample_time=0.05,
    causal_gates=None,
    encoder_time_features=None,
    trunk_time_features=None,
):
    y_pred = predict_mimo_causal_init(
        params,
        x_inputs,
        x_time,
        x_initial,
        branch_ranges,
        p,
        n_y,
        tau,
        sample_time,
        causal_gates,
        encoder_time_features,
        trunk_time_features,
    )
    return jnp.mean((y_pred - y_true) ** 2)


@partial(jax.jit, static_argnames=("branch_ranges", "p", "n_y", "piml_weight", "tau"))
def piml_mse_loss(
    params,
    x_inputs,
    x_time,
    x_initial,
    y_true,
    branch_ranges,
    p,
    n_y,
    piml_weight,
    tau=1.0,
    sample_time=0.05,
    causal_gates=None,
    encoder_time_features=None,
    trunk_time_features=None,
):
    y_pred_model = predict_mimo_causal_init(
        params,
        x_inputs,
        x_time,
        x_initial,
        branch_ranges,
        p,
        n_y,
        tau,
        sample_time,
        causal_gates,
        encoder_time_features,
        trunk_time_features,
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
    branch_ranges,
    p,
    n_y,
    loss_fn=mse_loss,
    piml_weight=None,
    tau=1.0,
    sample_time=0.05,
):
    @jax.jit
    def train_step(
        params,
        opt_state,
        x_inputs,
        x_time,
        x_initial,
        y_true,
        causal_gates,
        encoder_time_features,
        trunk_time_features,
    ):
        loss_value, grads = jax.value_and_grad(loss_fn)(
            params,
            x_inputs,
            x_time,
            x_initial,
            y_true,
            branch_ranges,
            p,
            n_y,
            piml_weight,
            tau,
            sample_time,
            causal_gates,
            encoder_time_features,
            trunk_time_features,
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
    decay_rate=0.95,
    transition_steps=200,
    l2_reg=1e-6,
    piml_weight=None,
    tau=1.0,
    sample_time=0.05,
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
        branch_ranges,
        p,
        n_y,
        loss_fn,
        piml_weight=piml_weight,
        tau=tau,
        sample_time=sample_time,
    )
    causal_features = precompute_causal_features(
        np.asarray(data["time_train"][:1]),
        len(branch_ranges),
        tau=tau,
        sample_time=sample_time,
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
            causal_features.gates,
            causal_features.encoder_times,
            causal_features.trunk_times,
        )
        losses.append(float(loss_value))
        if epoch % transition_steps == 0:
            print(f"Epoch: {epoch}/{epochs} with RMSE loss: {float(np.sqrt(loss_value))}")

    return params, losses


def evaluate_mimo_model_init(
    params, data, branch_ranges, p, n_y, tau=1.0, sample_time=0.05
):
    data_len = data["inputs_train"].shape[0]
    test_len = min(1000, int(data_len) - 1)
    if test_len <= 0:
        raise ValueError("Need at least two supervised samples to evaluate the model.")
    start = np.random.randint(0, int(data_len) - test_len + 1)
    x_time = data["time_train"][start : start + test_len]
    causal_features = precompute_causal_features(
        np.asarray(x_time[:1]),
        len(branch_ranges),
        tau=tau,
        sample_time=sample_time,
    )

    y_pred = predict_mimo_causal_init(
        params,
        data["inputs_train"][start : start + test_len],
        x_time,
        data["initial_train"][start : start + test_len],
        branch_ranges,
        p,
        n_y,
        tau,
        sample_time,
        causal_features.gates,
        causal_features.encoder_times,
        causal_features.trunk_times,
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
        "branch_ranges": np.asarray(bundle.branch_ranges, dtype=np.int32),
        "p": np.asarray(bundle.p, dtype=np.int32),
        "n_x": np.asarray(bundle.n_x, dtype=np.int32),
        "n_u": np.asarray(bundle.n_u, dtype=np.int32),
        "tau": np.asarray(bundle.tau, dtype=np.float32),
        "alpha": np.asarray(bundle.tau, dtype=np.float32),
        "horizon": np.asarray(bundle.horizon, dtype=np.int32),
        "sample_time": np.asarray(bundle.sample_time, dtype=np.float32),
        "n_encoders": np.asarray(len(bundle.params["encoders"]), dtype=np.int32),
        "n_encoder_layers": np.asarray(len(bundle.params["encoders"][0]), dtype=np.int32),
        "n_branch_layers": np.asarray(len(bundle.params["branch"]), dtype=np.int32),
        "n_trunk_layers": np.asarray(len(bundle.params["trunk"]), dtype=np.int32),
        "n_root_layers": np.asarray(0, dtype=np.int32),
    }
    for encoder_idx, encoder in enumerate(bundle.params["encoders"]):
        for layer_idx, layer in enumerate(encoder):
            arrays[_layer_key(f"encoder_{encoder_idx}", layer_idx, "w")] = np.asarray(
                layer["w"], dtype=np.float32
            )
            arrays[_layer_key(f"encoder_{encoder_idx}", layer_idx, "b")] = np.asarray(
                layer["b"], dtype=np.float32
            )
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
        n_encoders = int(data["n_encoders"])

        if n_encoders != 1:
            raise ValueError("deeponet_transf.load_bundle expected exactly one shared encoder, " f"but found {n_encoders}")
        
        n_encoder_layers = int(data["n_encoder_layers"])
        
        encoders = []
        for encoder_idx in range(n_encoders):
            layers = []
            for layer_idx in range(n_encoder_layers):
                layers.append(
                    {
                        "w": jnp.asarray(
                            data[_layer_key(f"encoder_{encoder_idx}", layer_idx, "w")]
                        ),
                        "b": jnp.asarray(
                            data[_layer_key(f"encoder_{encoder_idx}", layer_idx, "b")]
                        ),
                    }
                )
            encoders.append(tuple(layers))

        n_root_layers = int(data["n_root_layers"]) if "n_root_layers" in data.files else 0
        if n_root_layers:
            raise ValueError("deeponet_transf.load_bundle expected a rootless bundle")

        params = {"encoders": tuple(encoders)}
        n_u = int(data["n_u"])
        encoder_input_dim = int(params["encoders"][0][0]["w"].shape[0])
        if encoder_input_dim != n_u + 1:
            raise ValueError(
                "deeponet_transf.load_bundle expected encoder inputs [u(t_i), i*T_s] "
                f"with width {n_u + 1}, but found width {encoder_input_dim}"
            )
        for group_name in ("branch", "trunk"):
            n_layers = int(data[f"n_{group_name}_layers"])
            params[group_name] = tuple(
                {
                    "w": jnp.asarray(data[_layer_key(group_name, layer_idx, "w")]),
                    "b": jnp.asarray(data[_layer_key(group_name, layer_idx, "b")]),
                }
                for layer_idx in range(n_layers)
            )

        tau = float(data["tau"]) if "tau" in data.files else float(data["alpha"])
        return DeepONetBundle(
            params=params,
            branch_ranges=tuple(
                map(tuple, np.asarray(data["branch_ranges"], dtype=np.int32).tolist())
            ),
            p=int(data["p"]),
            n_x=int(data["n_x"]),
            n_u=n_u,
            tau=tau,
            horizon=int(data["horizon"]),
            sample_time=(
                float(data["sample_time"]) if "sample_time" in data.files else 0.05
            ),
        )


def predict_states(bundle: DeepONetBundle, x0: jnp.ndarray, u_seq: jnp.ndarray) -> jnp.ndarray:
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    u_seq = jnp.asarray(u_seq, dtype=jnp.float32)
    x_inputs = u_seq.reshape(1, -1)
    x_time = jnp.arange(1, bundle.horizon + 1, dtype=jnp.float32).reshape(1, -1)
    causal_features = standard_causal_features(
        bundle.horizon,
        bundle.tau,
        bundle.sample_time,
    )
    return predict_mimo_causal_init(
        bundle.params,
        x_inputs,
        x_time,
        x0.reshape(1, -1),
        bundle.branch_ranges,
        bundle.p,
        bundle.n_x,
        bundle.tau,
        bundle.sample_time,
        causal_features.gates,
        causal_features.encoder_times,
        causal_features.trunk_times,
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


def default_model_path(str="auto_car_learning/DeepONet_ARC_transf_default.npz") -> Path:
    return REPO_ROOT / "model" / str
