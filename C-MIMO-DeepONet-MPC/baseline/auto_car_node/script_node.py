"""Physics-informed neural one-step dynamics for the AutoCar data.

This baseline replaces the MIMO DeepONet with a shared, autoregressive neural
dynamics model

    x[k + 1] = x[k] + NN(x[k], u[k]).

The network predicts the direct state increment ``[dX, dY, dphi, dv]``.  In
particular, ``phi`` remains unwrapped throughout the autoregressive rollout.

The network is trained on one-step transitions from the smooth measured
trajectory.  The PIML term biases each transition toward the known discrete
physics model ``f_phys`` while the data term fits the measured next state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.auto_car_deeponet.script_jong2025 import load_arc_data  # noqa: E402
from model.auto_car.auto_car import f_phys  # noqa: E402


@dataclass(frozen=True)
class NODEBundle:
    params: tuple[dict, ...]
    n_x: int
    n_u: int
    horizon: int


def init_dense(key, in_dim, out_dim):
    """Xavier-uniform dense layer."""
    limit = jnp.sqrt(6.0 / float(in_dim + out_dim))
    return {
        "w": jax.random.uniform(
            key, (in_dim, out_dim), minval=-limit, maxval=limit,
            dtype=jnp.float32,
        ),
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def init_node_params(key, n_x, n_u, hidden_dim=96, num_hidden_layers=4):
    """Initialize NN(x, u) -> [dX, dY, dphi, dv]."""
    if n_x != 4:
        raise ValueError("The NODE expects state [X, Y, phi, v].")
    output_dim = n_x
    widths = [n_x + n_u] + [hidden_dim] * num_hidden_layers + [output_dim]
    keys = jax.random.split(key, len(widths) - 1)
    params = tuple(
        init_dense(layer_key, in_dim, out_dim)
        for layer_key, in_dim, out_dim in zip(keys, widths[:-1], widths[1:])
    )
    return params


def node_increment(params, x_k, u_k):
    """Evaluate NN(x[k], u[k]) as the direct state increment."""
    value = jnp.concatenate((x_k, u_k), axis=-1)
    for layer in params[:-1]:
        value = jax.nn.silu(value @ layer["w"] + layer["b"])
    last = params[-1]
    return value @ last["w"] + last["b"]


def node_step(params, x_k, u_k):
    """Apply the residual transition x[k + 1] = x[k] + NN(x[k], u[k])."""
    return x_k + node_increment(params, x_k, u_k)


@jax.jit
def predict_states_batch(params, x_initial, u_seq):
    """Autoregressively roll out a batch of control sequences."""

    def scan_step(x_k, u_k):
        x_next = node_step(params, x_k, u_k)
        return x_next, x_next

    _, states_time_major = jax.lax.scan(
        scan_step, x_initial, jnp.swapaxes(u_seq, 0, 1)
    )
    return jnp.swapaxes(states_time_major, 0, 1)


@jax.jit
def loss_components(params, x_current, u_current, x_next):
    """Return one-step data and physics losses over independent transitions."""
    y_pred = node_step(params, x_current, u_current)
    y_phys = jax.vmap(f_phys)(x_current, u_current)
    data_loss = jnp.mean(jnp.square(y_pred - x_next))
    physics_loss = jnp.mean(jnp.square(y_pred - y_phys))
    return data_loss, physics_loss


def make_train_step(optimizer, piml_weight):
    @jax.jit
    def train_step(params, opt_state, x_current, u_current, x_next):
        def objective(model_params):
            data_loss, physics_loss = loss_components(
                model_params, x_current, u_current, x_next
            )
            return data_loss + piml_weight * physics_loss, (
                data_loss,
                physics_loss,
            )

        (loss, components), grads = jax.value_and_grad(
            objective, has_aux=True
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, components

    return train_step


def train_node(
    params,
    x_current,
    u_current,
    x_next,
    *,
    epochs=20000,
    lr=1e-2,
    decay_rate=0.96,
    transition_steps=200,
    l2_reg=1e-6,
    piml_weight=0.1,
):
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        staircase=True,
    )
    optimizer = optax.chain(
        optax.add_decayed_weights(l2_reg), optax.adam(schedule)
    )
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer, piml_weight)
    history = []

    for epoch in range(epochs):
        params, opt_state, loss, (data_loss, physics_loss) = train_step(
            params, opt_state, x_current, u_current, x_next
        )
        history.append(float(loss))
        if epoch % transition_steps == 0:
            print(
                f"Epoch {epoch}/{epochs}: total RMSE={float(jnp.sqrt(loss)):.6f}, "
                f"data RMSE={float(jnp.sqrt(data_loss)):.6f}, "
                f"physics RMSE={float(jnp.sqrt(physics_loss)):.6f}"
            )
    return params, history


def make_one_step_transitions(x_data, u_data, lap_data):
    """Build (x[k], u[k], x[k + 1]) samples without crossing lap boundaries."""
    x_current = []
    u_current = []
    x_next = []
    lap_start = 0
    for count in np.asarray(lap_data, dtype=np.int64):
        lap_end = lap_start + int(count)
        x_current.append(x_data[lap_start : lap_end - 1])
        u_current.append(u_data[lap_start : lap_end - 1])
        x_next.append(x_data[lap_start + 1 : lap_end])
        lap_start = lap_end

    return tuple(
        jnp.asarray(np.concatenate(values, axis=0), dtype=jnp.float32)
        for values in (x_current, u_current, x_next)
    )


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "n_x": np.asarray(bundle.n_x, dtype=np.int32),
        "n_u": np.asarray(bundle.n_u, dtype=np.int32),
        "horizon": np.asarray(bundle.horizon, dtype=np.int32),
        "n_layers": np.asarray(len(bundle.params), dtype=np.int32),
    }
    for index, layer in enumerate(bundle.params):
        arrays[f"layer_{index}_w"] = np.asarray(layer["w"], dtype=np.float32)
        arrays[f"layer_{index}_b"] = np.asarray(layer["b"], dtype=np.float32)
    np.savez(path, **arrays)
    return path


def load_bundle(path):
    with np.load(path) as archive:
        params = tuple(
            {
                "w": jnp.asarray(archive[f"layer_{i}_w"]),
                "b": jnp.asarray(archive[f"layer_{i}_b"]),
            }
            for i in range(int(archive["n_layers"]))
        )
        n_x = int(archive["n_x"])
        if params[-1]["b"].shape[0] != n_x:
            raise ValueError(
                "This checkpoint does not use the residual four-state NODE "
                "output. Retrain it with x[k + 1] = x[k] + NN(x[k], u[k])."
            )
        return NODEBundle(
            params=params,
            n_x=n_x,
            n_u=int(archive["n_u"]),
            horizon=int(archive["horizon"]),
        )


def predict_states(bundle, x0, u_seq):
    """MPC-friendly unbatched rollout helper."""
    x0 = jnp.asarray(x0, dtype=jnp.float32).reshape(1, bundle.n_x)
    u_seq = jnp.asarray(u_seq, dtype=jnp.float32).reshape(
        1, bundle.horizon, bundle.n_u
    )
    return predict_states_batch(bundle.params, x0, u_seq)[0]


def predict_states_u_jacobian(bundle, x0, u_seq):
    """Jacobian shaped (horizon, n_x, horizon, n_u) for MPC."""
    u_seq = jnp.asarray(u_seq, dtype=jnp.float32)
    return jax.jacfwd(lambda controls: predict_states(bundle, x0, controls))(u_seq)


def main():
    print("JAX devices:", jax.devices())
    data_name = "AC_PP_50000_07271410"
    horizon = 20
    hidden_dim = 96
    hidden_layers = 4
    epochs = 30000
    split_seed = 7
    transition_steps = 300
    lr = 5e-3
    decay_rate = 0.96
    l2_reg = 1e-6
    piml_weight = 1.0

    data_file = REPO_ROOT / "data" / "training" / f"{data_name}.npz"
    x_data, u_data, lap_data = load_arc_data(data_file)
    n_x, n_u = x_data.shape[1], u_data.shape[1]
    x_current, u_current, x_next = make_one_step_transitions(
        x_data, u_data, lap_data
    )

    print("one-step training states:", x_current.shape)
    print("one-step training controls:", u_current.shape)
    print("one-step training targets:", x_next.shape)
    print(f"n_x={n_x}, n_u={n_u}, PIML weight={piml_weight}")

    params = init_node_params(
        jax.random.PRNGKey(split_seed), n_x, n_u,
        hidden_dim=hidden_dim, num_hidden_layers=hidden_layers,
    )
    params, _ = train_node(
        params, x_current, u_current, x_next,
        epochs=epochs, lr=lr, decay_rate=decay_rate,
        transition_steps=transition_steps, l2_reg=l2_reg,
        piml_weight=piml_weight,
    )

    model_path = (
        REPO_ROOT / "model" / "auto_car_learning" /
        f"PIML_NODE_{data_name}_h={horizon}_hidden={hidden_dim}.npz"
    )
    bundle = NODEBundle(params=params, n_x=n_x, n_u=n_u, horizon=horizon)
    saved_path = save_bundle(bundle, model_path)
    print(f"Saved PIML Neural ODE model to {saved_path}")


if __name__ == "__main__":
    main()
