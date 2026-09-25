"""
The batch-size-scaling study's timing (#367; results in docs/optimizations/current-baseline.md
and candidates.md): dense 784 -> 30 -> 10 epoch timing on full MNIST by batch size, numpy
against Rust, plus a Rust op profile.

    python scripts/batch_size_timing.py time [--repeats 5] [--batch-sizes 32 128 512 1024]
    python scripts/batch_size_timing.py profile [--batch-sizes 512 1024]

`time` runs every (backend, batch size, repeat) in its own process, never two backends in one,
rotating the order each repeat, and reports medians. Each process starts from the same weights
(numpy-drawn from seed 0) with random.seed(0) before each epoch, and measures:

- epoch: one epoch of train_backprop_network_mini_batch, the trainer the demos use, including
  its two training-set accuracy passes (before and after the epoch, for the pocket snapshot);
- steps: the learn_batch calls alone, over the same epoch;
- accuracy pass: one _training_accuracy pass over the 60000 training rows;
- batch conversion: turning every batch of the epoch from tuples into the backend's array, the
  first line of learn_batch;
- row conversion: turning every training row into a backend array, as each classify_state of the
  accuracy pass does.

The rate is plain SGD at the scaled lr_32 = 4 with a one-epoch warmup, a setting stable in the scaling sweep;
the rate doesn't change the work per step. numpy runs with its default OpenBLAS threading.

`profile` is the dense counterpart of the conv demo's rust_op_breakdown: cProfile over one Rust
step loop, own time by indrajala_math_rust op. cProfile's overhead inflates the Python side, so
these are shares of a profiled run, not of the timed runs.
"""

import argparse
import cProfile
import json
import pstats
import random
import statistics
import sys
import time

import indrajala_math_rust as pa
import numpy as np
from process_runs import interleaved_runs, run_json_worker

from indrajala_ml import batch_size_scaling as bss
from indrajala_ml.demos.demo_conv_rust_vs_vectorized_digit_recognition import _rust_op_name
from indrajala_ml.mnist_data import load_mnist_dataset
from indrajala_ml.train import _training_accuracy, train_backprop_network_mini_batch

BACKENDS = ["numpy", "rust"]
BATCH_SIZES = [32, 128, 512, 1024]
PROFILE_BATCH_SIZES = [512, 1024]
BASE_RATE = 4.0
WARMUP_EPOCHS = 1.0
SEED = 0
MEASURES = ["epoch", "steps", "accuracy pass", "batch conversion", "row conversion"]


def _schedule(train_size: int, batch_size: int):
    rate = bss.scaled_learning_rate(BASE_RATE, batch_size)
    return bss.learning_rate_schedule(rate, bss.warmup_steps(WARMUP_EPOCHS, train_size, batch_size))


def _to_array(backend: str):
    return np.array if backend == "numpy" else pa.Array


def measure(backend: str, batch_size: int, train_data: list) -> dict:
    schedule = _schedule(len(train_data), batch_size)
    to_array = _to_array(backend)

    network = bss.initial_network(backend, 0.0, SEED)
    random.seed(SEED)
    start = time.perf_counter()
    train_backprop_network_mini_batch(network, train_data, batch_size, learning_rate=schedule, epochs=1)
    epoch = time.perf_counter() - start

    network = bss.initial_network(backend, 0.0, SEED)
    random.seed(SEED)
    _steps, steps = bss.train_epoch(network, train_data, batch_size, schedule, first_step=0)

    start = time.perf_counter()
    _training_accuracy(network, train_data)
    accuracy_pass = time.perf_counter() - start

    batches = [train_data[i : i + batch_size] for i in range(0, len(train_data), batch_size)]
    start = time.perf_counter()
    for batch in batches:
        to_array([list(state) for state, _label in batch])
    batch_conversion = time.perf_counter() - start

    start = time.perf_counter()
    for state, _label in train_data:
        to_array(list(state))
    row_conversion = time.perf_counter() - start

    return {
        "epoch": epoch,
        "steps": steps,
        "accuracy pass": accuracy_pass,
        "batch conversion": batch_conversion,
        "row conversion": row_conversion,
    }


def profile(batch_size: int, train_data: list) -> tuple[float, float, list]:
    """(step-loop seconds unprofiled, total profiled seconds, [(op, seconds, calls)]), largest first."""
    schedule = _schedule(len(train_data), batch_size)

    network = bss.initial_network("rust", 0.0, SEED)
    random.seed(SEED)
    _steps, steps = bss.train_epoch(network, train_data, batch_size, schedule, first_step=0)

    network = bss.initial_network("rust", 0.0, SEED)
    random.seed(SEED)
    profiler = cProfile.Profile()
    profiler.enable()
    bss.train_epoch(network, train_data, batch_size, schedule, first_step=0)
    profiler.disable()

    stats = pstats.Stats(profiler)
    ops = []
    for (_file, _line, name), (_calls, total_calls, own_seconds, _cumulative, _callers) in stats.stats.items():
        op = _rust_op_name(name)
        if op is not None:
            ops.append((op, own_seconds, total_calls))
    return steps, stats.total_tt, sorted(ops, key=lambda op: op[1], reverse=True)


def _run_worker(backend: str, batch_size: int) -> dict:
    return run_json_worker([sys.executable, __file__, "worker", backend, str(batch_size)])


def time_all(batch_sizes: list[int], repeats: int) -> dict:
    cells = [(backend, batch_size) for batch_size in batch_sizes for backend in BACKENDS]
    runs = interleaved_runs(cells, repeats, lambda cell: _run_worker(*cell))

    medians = {
        cell: {m: statistics.median(run[m] for run in cell_runs) for m in MEASURES} for cell, cell_runs in runs.items()
    }

    print(f"median of {repeats}, seconds per epoch (one process per measurement)\n")
    print(
        "| B | backend | epoch | steps | accuracy pass | batch conversion | row conversion | steps % of epoch | accuracy passes % of epoch |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for batch_size in batch_sizes:
        for backend in BACKENDS:
            m = medians[(backend, batch_size)]
            print(
                f"| {batch_size} | {backend} | {m['epoch']:.2f} | {m['steps']:.2f} | {m['accuracy pass']:.2f} | "
                f"{m['batch conversion']:.3f} | {m['row conversion']:.2f} | {m['steps'] / m['epoch']:.0%} | "
                f"{2 * m['accuracy pass'] / m['epoch']:.0%} |"
            )
    print("\nRust / numpy ratio of medians\n")
    print("| B | epoch | steps | accuracy pass |")
    print("|---|---|---|---|")
    for batch_size in batch_sizes:
        numpy_m, rust_m = medians[("numpy", batch_size)], medians[("rust", batch_size)]
        print(
            f"| {batch_size} | {rust_m['epoch'] / numpy_m['epoch']:.2f} | {rust_m['steps'] / numpy_m['steps']:.2f} | "
            f"{rust_m['accuracy pass'] / numpy_m['accuracy pass']:.2f} |"
        )
    return {f"{backend} {batch_size}": cell_runs for (backend, batch_size), cell_runs in runs.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["time", "profile", "worker"])
    parser.add_argument("args", nargs="*", help="worker only: backend batch_size")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", help="time only: write every run's raw measurements here as JSON")
    args = parser.parse_args(argv)

    if args.mode == "worker":
        backend, batch_size = args.args[0], int(args.args[1])
        train_data = load_mnist_dataset(bss.TRAIN_PATH)
        print(json.dumps(measure(backend, batch_size, train_data)))
    elif args.mode == "time":
        runs = time_all(args.batch_sizes or BATCH_SIZES, args.repeats)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(runs, f, indent=1)
    else:
        train_data = load_mnist_dataset(bss.TRAIN_PATH)
        for batch_size in args.batch_sizes or PROFILE_BATCH_SIZES:
            steps, total, ops = profile(batch_size, train_data)
            op_total = sum(seconds for _op, seconds, _calls in ops)
            print(
                f"\n### B = {batch_size}: step loop {steps:.2f} s unprofiled, {total:.2f} s profiled, Rust ops {op_total:.2f} s\n"
            )
            print("| op | s | calls | ms / call | % of profiled |")
            print("|---|---|---|---|---|")
            for op, seconds, calls in ops:
                print(f"| {op} | {seconds:.3f} | {calls} | {1000 * seconds / calls:.3f} | {seconds / total:.1%} |")


if __name__ == "__main__":
    main()
