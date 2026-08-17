"""Benchmark CATR-MIMO DeepONet terminal inference and control Jacobians."""

from __future__ import annotations

from functools import partial
from pathlib import Path
import platform
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deeponet import ( 
    load_arc_data,
    load_bundle,
    make_mimo_supervised_sequences,
    precompute_causal_features,
    predict_mimo_causal_init,
    )


from scripts.auto_car_train_sim.catr_mimo_deeponet import (
    DATA_NAME,
    HORIZON,
    P,
    TAU,
    ENCODE_HIDDEN,
    BRANCH_HIDDEN,
    TRUNK_HIDDEN
)

BENCHMARK_SAMPLES = 1000
BENCHMARK_SEED = 5
WARMUP_ITERATIONS = 2

def terminal_state(
    params,
    x0,
    u_seq,
    *,
    branch_ranges,
    p,
    n_x,
    tau,
    sample_time,
    causal_gates,
    encoder_time_features,
    trunk_time_features,
):
    """Compute x[H|k] for one initial state and one control sequence."""
    x_inputs = u_seq.reshape(1, -1)
    x_time = jnp.asarray([[u_seq.shape[0]]], dtype=jnp.float32)
    prediction = predict_mimo_causal_init(
        params,
        x_inputs,
        x_time,
        x0.reshape(1, -1),
        branch_ranges,
        p,
        n_x,
        tau,
        sample_time,
        causal_gates,
        encoder_time_features,
        trunk_time_features,
    )
    return prediction[0, 0]


def terminal_full_u_jacobian(
    params,
    x0,
    u_seq,
    *,
    branch_ranges,
    p,
    n_x,
    tau,
    sample_time,
    causal_gates,
    encoder_time_features,
    trunk_time_features,
):
    """Compute d x[H|k] / d u[0:H-1|k], shape (n_x, H, n_u)."""
    return jax.jacfwd(
        lambda controls: terminal_state(
            params,
            x0,
            controls,
            branch_ranges=branch_ranges,
            p=p,
            n_x=n_x,
            tau=tau,
            sample_time=sample_time,
            causal_gates=causal_gates,
            encoder_time_features=encoder_time_features,
            trunk_time_features=trunk_time_features,
        )
    )(u_seq)


def random_benchmark_inputs(bundle):
    """Select reproducible random H-step windows from the training trajectory."""
    data_path = REPO_ROOT / "data" / "training" / f"{DATA_NAME}.npz"
    x_data, u_data, lap_data, _ = load_arc_data(data_path)
    data = make_mimo_supervised_sequences(
        x_data,
        u_data,
        lap_data,
        10_000,
        20_000,
        num_pred=bundle.horizon,
        split_seed=BENCHMARK_SEED,
    )
    all_x0 = np.asarray(data["initial_train"], dtype=np.float32)
    all_u = np.asarray(data["inputs_train"], dtype=np.float32).reshape(
        -1, bundle.horizon, bundle.n_u
    )
    count = min(BENCHMARK_SAMPLES, all_x0.shape[0])
    indices = np.random.default_rng(BENCHMARK_SEED).choice(
        all_x0.shape[0], size=count, replace=False
    )
    return all_x0[indices], all_u[indices]


def sample_latencies_ms(compiled_function, params, x0, u_seq):
    """Measure synchronized latency for every benchmark trajectory."""
    latencies = np.empty(x0.shape[0], dtype=np.float64)
    for sample in range(x0.shape[0]):
        x0_sample = x0[sample]
        u_sample = u_seq[sample]
        x0_sample.block_until_ready()
        u_sample.block_until_ready()

        start_ns = time.perf_counter_ns()
        compiled_function(params, x0_sample, u_sample).block_until_ready()
        latencies[sample] = (time.perf_counter_ns() - start_ns) / 1_000_000

    return latencies


def warmup_all(functions, params, x0, u_seq):
    """Compile and repeatedly warm every operation before any timing starts."""
    output_shapes = {}
    for iteration in range(WARMUP_ITERATIONS):
        sample = iteration % x0.shape[0]
        x0_sample = x0[sample]
        u_sample = u_seq[sample]
        x0_sample.block_until_ready()
        u_sample.block_until_ready()
        for name, compiled_function in functions.items():
            result = compiled_function(params, x0_sample, u_sample)
            result.block_until_ready()
            output_shapes[name] = result.shape
    return output_shapes


def device_name(device):
    """Return a descriptive CPU or accelerator model name."""
    if device.platform != "cpu":
        return device.device_kind

    processor_name = ""
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                processor_name = line.split(":", maxsplit=1)[1].strip()
                break
    if not processor_name:
        processor_name = platform.processor().strip()
    return processor_name or device.device_kind


def benchmark_device(device, bundle, params_host, x0_host, u_host):
    """Compile and benchmark all CATR-MIMO operations on one device."""
    params = jax.device_put(params_host, device)
    x0 = jax.device_put(x0_host, device)
    u_seq = jax.device_put(u_host, device)
    jax.block_until_ready((params, x0, u_seq))

    causal_features = precompute_causal_features(
        np.asarray([bundle.horizon], dtype=np.float32),
        bundle.horizon,
        tau=bundle.tau,
        sample_time=bundle.sample_time,
    )

    model_arguments = {
        "branch_ranges": bundle.branch_ranges,
        "p": bundle.p,
        "n_x": bundle.n_x,
        "tau": bundle.tau,
        "sample_time": bundle.sample_time,
        "causal_gates": causal_features.gates,
        "encoder_time_features": causal_features.encoder_times,
        "trunk_time_features": causal_features.trunk_times,
    }

    inference = partial(terminal_state, **model_arguments)

    full_jacobian = partial(terminal_full_u_jacobian, **model_arguments)

    functions = {
        "inference": jax.jit(inference),
        "full_jacobian": jax.jit(full_jacobian),
    }

    output_shapes = warmup_all(functions, params, x0, u_seq)

    sample_times = {}
    rng = np.random.default_rng(BENCHMARK_SEED)

    # Randomizing the order prevents one operation from consistently benefiting
    # from device clock ramp-up or always running first.
    for name in rng.permutation(tuple(functions)):
        sample_times[name] = sample_latencies_ms(
            functions[name], params, x0, u_seq
        )

    mean_times = {
        name: float(np.mean(measurements))
        for name, measurements in sample_times.items()
    }

    std_times = {
        name: float(np.std(measurements))
        for name, measurements in sample_times.items()
    }

    print(f"\n{device.platform.upper()} benchmark: {device_name(device)}")
    print(f"Trajectories: {x0.shape[0]}, horizon: {u_seq.shape[1]}")
    print(f"Warm-up iterations: {WARMUP_ITERATIONS}")
    print(
        f"Average x[H|k] inference time ({output_shapes['inference']}): "
        f"{mean_times['inference']:.6f} +/- "
        f"{std_times['inference']:.6f} ms"
    )
    print(
        "Average d x[H|k] / d u[0:H-1|k] time "
        f"({output_shapes['full_jacobian']}): "
        f"{mean_times['full_jacobian']:.6f} +/- "
        f"{std_times['full_jacobian']:.6f} ms"
    )


def available_benchmark_devices():
    """Return one CPU and, when available, one GPU device."""
    devices = [("CPU", jax.devices("cpu")[0])]
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    if gpu_devices:
        devices.append(("GPU", gpu_devices[0]))
    return devices


def evaluate_FDE(bundle):

    params = bundle.params
    branch_ranges = bundle.branch_ranges
    p = bundle.p
    n_y = bundle.n_x
    tau = bundle.tau
    sample_time = bundle.sample_time

    data_path = REPO_ROOT / "data" / "training" / f"{DATA_NAME}.npz"
    x_data, u_data, lap_data, sample_time = load_arc_data(data_path)

    data = make_mimo_supervised_sequences(
        x_data,
        u_data,
        lap_data,
        10_000,
        20_000,
        num_pred=bundle.horizon,
        split_seed=BENCHMARK_SEED,
    )

    data_len = data["inputs_train"].shape[0]
    test_len = min(1000, int(data_len) - 1)

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

    H = y_pred.shape[1]

    FDE = np.mean(np.sqrt((y_pred[:, H-1, 0] - y_true[:, H-1, 0]) ** 2 + (y_pred[:, H-1, 1] - y_true[:, H-1, 1]) ** 2))
    ADE = np.mean(np.sqrt((y_pred[:, :, 0]   - y_true[:, :, 0]) ** 2   + (y_pred[:, :, 1]   - y_true[:, :, 1]) ** 2))

    return FDE, ADE


def main():
    model_path = (
        REPO_ROOT / "model" / "auto_car_learning"
        / f"CATR_MIMO_DeepONet_{DATA_NAME}_p={P}_b={TAU:g}"
        f"_e={ENCODE_HIDDEN}_br={BRANCH_HIDDEN}_tr={TRUNK_HIDDEN}.npz"
    )
    bundle = load_bundle(model_path)

    print(f"Loaded CATR-MIMO DeepONet model from {model_path}")


    FDE, ADE = evaluate_FDE(bundle)

    print(f"Final Displacement Error = {FDE}")
    print(f"Average Displacement Error = {ADE}")


    x0_host, u_host = random_benchmark_inputs(bundle)
    params_host = jax.tree_util.tree_map(
        np.asarray, jax.device_get(bundle.params)
    )
    print(
        f"Randomly selected {x0_host.shape[0]} H-step trajectories "
        f"with seed {BENCHMARK_SEED}"
    )
    print("JIT compilation and host-to-device transfer are excluded.")

    devices = available_benchmark_devices()
    for _, device in devices:
        benchmark_device(device, bundle, params_host, x0_host, u_host)
    if not any(label == "GPU" for label, _ in devices):
        print("\nGPU benchmark skipped: no JAX GPU backend is available.")


if __name__ == "__main__":
    main()
