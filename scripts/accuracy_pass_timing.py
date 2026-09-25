# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, which strict mode takes for constants)
"""
The batched accuracy pass's stage 0 (docs/optimizations/implemented.md): one training-set
accuracy pass row by row, as _training_accuracy does it, against batched forward passes over
chunks of the prepared matrix.

    python scripts/accuracy_pass_timing.py time [--repeats 5] [--networks ...] [--out runs.json]
    python scripts/accuracy_pass_timing.py report runs.json

Every (network, backend, repeat) runs in its own process, never two backends in one, rotating
the order each repeat, and the medians are reported. Inputs are pre-converted: the dataset is
prepared_mnist's, so no pass converts a row.

Networks, from numpy-drawn seed-0 weights (as scripts/prepared_dataset_timing.py):
- dense: 784 -> 30 -> 10 on full MNIST (60000 rows);
- conv, conv-pool-conv, conv-conv-stride2: the conv demo's architectures (dense 32) on its
  2000-row MNIST subset.

Measures (seconds), each the median of 3 in-process runs:
- per row: the classify_row loop _training_accuracy runs;
- forward B: forward_batch over chunks of B rows (a take_rows copy per chunk in Rust, a view in
  numpy), without the argmax;
- batched B: the same plus the argmax per row (np.argmax(axis=1) in numpy, tolist and a Python
  argmax per row in Rust, which has no row-wise argmax);
- epoch B=32 / epoch single: one trainer epoch given the prepared dataset (two accuracy passes
  included), from fresh weights, for the stake as a share of an epoch: of a one-epoch run (two
  passes) and of a long run (about one pass per epoch).

Each worker also counts the rows whose batched prediction differs from the per-row one.
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
from process_runs import interleaved_runs, run_json_worker

from indrajala_ml import batch_size_scaling as bss
from indrajala_ml.demos.demo_conv_rust_vs_vectorized_digit_recognition import ARCHITECTURES
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.prepared_dataset import PreparedDataset, prepared_mnist
from indrajala_ml.train import train_backprop_network_mini_batch, train_linear_classifier_network

BACKENDS = ["numpy", "rust"]
NETWORKS = ["dense", *ARCHITECTURES]
CHUNKS = [32, 512]
MEASURES = (
    ["per row"] + [f"forward {c}" for c in CHUNKS] + [f"batched {c}" for c in CHUNKS] + ["epoch B=32", "epoch single"]
)
LEARNING_RATE = 0.5
BATCH_SIZE = 32
SEED = 0
CONV_DENSE_LAYER_SIZES = [32]
CONV_TRAIN_LIMIT = 2000
CONV_CLASSES = {
    "numpy": ConvVectorizedMultiClassBackpropClassifierNetwork,
    "rust": ConvRustArrayMultiClassBackpropClassifierNetwork,
}
IN_PROCESS_RUNS = 3

T = TypeVar("T")

# the dense and conv networks of either backend
Network = VectorizedMultiClassBackpropClassifierNetwork | RustArrayMultiClassBackpropClassifierNetwork


def _network(name: str, backend: str) -> Network:
    if name == "dense":
        return bss.initial_network(backend, 0.0, SEED)
    np.random.seed(SEED)
    conv_specs = ARCHITECTURES[name]
    snapshot = ConvVectorizedMultiClassBackpropClassifierNetwork.randomized(
        28, 28, conv_specs, CONV_DENSE_LAYER_SIZES, 10
    ).snapshot()
    network = CONV_CLASSES[backend](28, 28, conv_specs, CONV_DENSE_LAYER_SIZES, 10)
    network.restore(snapshot)
    return network


def _per_row(network: Network, prepared: PreparedDataset) -> list[int]:
    return [network.classify_row(prepared, index) for index in range(len(prepared))]


def _forward_chunks(network: Network, prepared: PreparedDataset, chunk: int, backend: str, argmax: bool) -> list[int]:
    # either backend's matrix, chosen by name: its arrays are typed Any at that one boundary
    states: Any = prepared.states
    predictions: list[int] = []
    for start in range(0, len(prepared), chunk):
        stop = min(start + chunk, len(prepared))
        X: Any = states[start:stop] if backend == "numpy" else states.take_rows(list(range(start, stop)))
        for layer in network.layers:
            X = layer.forward_batch(X)
        if not argmax:
            continue
        if backend == "numpy":
            predictions.extend(np.argmax(X, axis=1).tolist())
        else:
            predictions.extend(row.index(max(row)) for row in X.tolist())
    return predictions


def _timed(fn: Callable[[], T]) -> tuple[float, T]:
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def _median_of_runs(fn: Callable[[], T]) -> tuple[float, T]:
    runs = [_timed(fn) for _ in range(IN_PROCESS_RUNS)]
    return statistics.median(seconds for seconds, _ in runs), runs[0][1]


def _epoch(name: str, backend: str, prepared: PreparedDataset, single: bool) -> float:
    network = _network(name, backend)
    random.seed(SEED)
    if single:
        return _timed(
            lambda: train_linear_classifier_network(network, prepared, learning_rate=LEARNING_RATE, epochs=1)
        )[0]
    return _timed(
        lambda: train_backprop_network_mini_batch(network, prepared, BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=1)
    )[0]


def measure(name: str, backend: str) -> dict[str, float]:
    limit = None if name == "dense" else CONV_TRAIN_LIMIT
    prepared = prepared_mnist(bss.TRAIN_PATH, backend, limit=limit)
    network = _network(name, backend)

    result: dict[str, float] = {}
    result["per row"], reference = _median_of_runs(lambda: _per_row(network, prepared))
    for chunk in CHUNKS:
        result[f"forward {chunk}"], _ = _median_of_runs(
            lambda: _forward_chunks(network, prepared, chunk, backend, False)
        )
        result[f"batched {chunk}"], predictions = _median_of_runs(
            lambda: _forward_chunks(network, prepared, chunk, backend, True)
        )
        result[f"mismatches {chunk}"] = sum(1 for a, b in zip(reference, predictions) if a != b)
    result["epoch B=32"] = _epoch(name, backend, prepared, single=False)
    result["epoch single"] = _epoch(name, backend, prepared, single=True)
    return result


def _run_worker(name: str, backend: str) -> dict[str, Any]:
    return run_json_worker([sys.executable, __file__, "worker", name, backend])


def time_all(networks: list[str], repeats: int) -> dict[str, list[dict[str, Any]]]:
    cells = [(name, backend) for name in networks for backend in BACKENDS]
    cell_runs_by_cell = interleaved_runs(cells, repeats, lambda cell: _run_worker(*cell))

    runs = {f"{name} / {backend}": cell_runs for (name, backend), cell_runs in cell_runs_by_cell.items()}
    report(runs)
    return runs


def report(runs: dict[str, list[dict[str, Any]]]) -> None:
    repeats = len(next(iter(runs.values())))
    medians = {
        cell: {m: statistics.median(run[m] for run in cell_runs) for m in MEASURES} for cell, cell_runs in runs.items()
    }

    print(f"median of {repeats} processes (each measure the median of {IN_PROCESS_RUNS} runs), seconds\n")
    print("| network / backend | " + " | ".join(MEASURES) + " |")
    print("|---|" + "---|" * len(MEASURES))
    for cell, median in medians.items():
        print(f"| {cell} | " + " | ".join(f"{median[m]:.3f}" for m in MEASURES) + " |")

    # a one-epoch run has two passes, its worst case; over a long run there is about one pass per
    # epoch, so the long-run share is one pass's saving against an epoch without one of its passes
    print("\nthe saving from batching, as a share of an epoch, and mismatched rows\n")
    print(
        "| network / backend | chunk | saved per pass | one-epoch B=32 | one-epoch single | long-run B=32 | long-run single | mismatches |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for cell, median in medians.items():
        for chunk in CHUNKS:
            saved = median["per row"] - median[f"batched {chunk}"]
            shares = [2 * saved / median[epoch] for epoch in ("epoch B=32", "epoch single")]
            shares += [saved / (median[epoch] - median["per row"]) for epoch in ("epoch B=32", "epoch single")]
            mismatches = max(run[f"mismatches {chunk}"] for run in runs[cell])
            print(
                f"| {cell} | {chunk} | {saved:.3f} | "
                + " | ".join(f"{share:.0%}" for share in shares)
                + f" | {mismatches} |"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["time", "worker", "report"])
    parser.add_argument("args", nargs="*", help="worker: network backend; report: runs.json")
    parser.add_argument("--networks", nargs="+", choices=NETWORKS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", help="write every run's raw measurements here as JSON")
    args = parser.parse_args(argv)

    if args.mode == "worker":
        print(json.dumps(measure(args.args[0], args.args[1])))
    elif args.mode == "report":
        with open(args.args[0]) as f:
            report(json.load(f))
    else:
        runs = time_all(args.networks or NETWORKS, args.repeats)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(runs, f, indent=1)


if __name__ == "__main__":
    main()
