"""Benchmark LU2019 unstacked DeepONet terminal inference and Jacobians."""

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

from baseline.auto_car_deeponet.script_jong2025 import (  # noqa: E402
    load_arc_data,
    load_bundle,
    make_mimo_supervised_sequences,
    predict_mimo_unstack_init,
)


DATA_NAME = "AC_PP_50000_07271410"
HORIZON = 20
P = 20
BRANCH_HIDDEN = 96
TRUNK_HIDDEN = 96
BENCHMARK_SAMPLES = 1000
BENCHMARK_SEED = 7
WARMUP_ITERATIONS = 2


def terminal_state(params, x0, u_seq, *, p, n_x):
    """Compute x[H|k] for one initial state and one control sequence."""
    x_inputs = u_seq.reshape(1, -1)
    x_time = jnp.asarray([[u_seq.shape[0]]], dtype=jnp.float32)
    prediction = predict_mimo_unstack_init(
        params,
        x_inputs,
        x_time,
        x0.reshape(1, -1),
        p,
        n_x,
    )
    return prediction[0, 0]


def terminal_full_u_jacobian(params, x0, u_seq, *, p, n_x):
    """Compute d x[H|k] / d u[0:H-1|k], shape (n_x, H, n_u)."""
    return jax.jacfwd(
        lambda controls: terminal_state(
            params, x0, controls, p=p, n_x=n_x
        )
    )(u_seq)


def random_benchmark_inputs(bundle):
    """Select reproducible random H-step windows from the training trajectory."""
    data_file = REPO_ROOT / "data" / "training" / f"{DATA_NAME}.npz"
    x_data, u_data, lap_data = load_arc_data(data_file)
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
    """Compile and benchmark all DeepONet operations on one explicit device."""
    params = jax.device_put(params_host, device)
    x0 = jax.device_put(x0_host, device)
    u_seq = jax.device_put(u_host, device)
    jax.block_until_ready((params, x0, u_seq))

    inference = partial(terminal_state, p=bundle.p, n_x=bundle.n_x)
    full_jacobian = partial(
        terminal_full_u_jacobian, p=bundle.p, n_x=bundle.n_x
    )
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


def main():
    model_path = (
        REPO_ROOT
        / "model"
        / "auto_car_learning"
        / f"PIML_DeepONet_unstack_baseline_{DATA_NAME}_p={P}"
        f"_br={BRANCH_HIDDEN}_tr={TRUNK_HIDDEN}.npz"
    )
    bundle = load_bundle(model_path)
    if bundle.horizon != HORIZON:
        raise ValueError(
            f"Expected horizon {HORIZON}, but model uses {bundle.horizon}."
        )
    print(f"Loaded LU2019 unstacked DeepONet model from {model_path}")

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
