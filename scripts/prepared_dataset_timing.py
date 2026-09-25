"""
The prepared dataset's A/B (docs/optimizations/implemented.md): one training epoch through the
trainers the demos use, before and after the dataset became one backend array (#373, #374).

    python scripts/prepared_dataset_timing.py time [--repeats 5] [--configs ...] [--epochs 1] [--out runs.json]

Every (config, backend, repeat) runs in its own process, never two backends in one, rotating the
order each repeat, and the medians are reported. The script only uses interfaces that exist on
both sides of the change, so the "before" numbers come from running this same file with an older
checkout first on PYTHONPATH (the output starts with the trainers module it imported):

    PYTHONPATH=/path/to/old/checkout python scripts/prepared_dataset_timing.py time

--epochs trains each run for more epochs (one accuracy pass per epoch, plus one before), as a
longer run does; the batched accuracy pass's A/B used it. Configs, each one epoch by default,
from numpy-drawn seed-0 weights with random.seed(0):
- dense B=32 / dense single: 784 -> 30 -> 10 on full MNIST (60000 rows), learning rate 0.5;
- conv B=32 / conv single: the conv demo's "conv" network (ConvSpec(3, 8), dense 32) on its
  2000-row MNIST subset, learning rate 0.5.

Measures (seconds):
- epoch: the trainer given the tuple list, as every demo calls it, including its two
  training-set accuracy passes (before and after the epoch, for the pocket snapshot); after the
  change this includes preparing the dataset once;
- prepare (after only): student.prepare_dataset of the tuple list alone;
- epoch, loader (after only): the trainer given prepared_mnist's dataset, which never builds
  Python floats; its load time is not included.
"""

import argparse
import json
import random
import statistics
import sys
import time

import numpy as np
from process_runs import interleaved_runs, run_json_worker

from indrajala_ml import batch_size_scaling as bss
from indrajala_ml import train
from indrajala_ml.mnist_data import load_mnist_dataset
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.train import train_backprop_network_mini_batch, train_linear_classifier_network

# indrajala_ml is a namespace package, so with an old checkout first on PYTHONPATH the new
# prepared_dataset module can still be imported from this one; the trainers themselves tell
# which side of the change is running
AFTER = hasattr(train, "_prepared_for")
if AFTER:
    from indrajala_ml.prepared_dataset import prepared_mnist

BACKENDS = ["numpy", "rust"]
CONFIGS = ["dense B=32", "dense single", "conv B=32", "conv single"]
MEASURES = ["epoch", "prepare", "epoch, loader"]
LEARNING_RATE = 0.5
BATCH_SIZE = 32
SEED = 0
CONV_SPECS = [ConvSpec(3, 8)]
CONV_DENSE_LAYER_SIZES = [32]
CONV_TRAIN_LIMIT = 2000
EPOCHS = 1  # --epochs
CONV_CLASSES = {
    "numpy": ConvVectorizedMultiClassBackpropClassifierNetwork,
    "rust": ConvRustArrayMultiClassBackpropClassifierNetwork,
}


def _network(config: str, backend: str):
    if config.startswith("dense"):
        return bss.initial_network(backend, 0.0, SEED)
    np.random.seed(SEED)
    snapshot = ConvVectorizedMultiClassBackpropClassifierNetwork.randomized(
        28, 28, CONV_SPECS, CONV_DENSE_LAYER_SIZES, 10
    ).snapshot()
    network = CONV_CLASSES[backend](28, 28, CONV_SPECS, CONV_DENSE_LAYER_SIZES, 10)
    network.restore(snapshot)
    return network


def _train_epoch(config: str, network, data) -> float:
    random.seed(SEED)
    start = time.perf_counter()
    if config.endswith("single"):
        train_linear_classifier_network(network, data, learning_rate=LEARNING_RATE, epochs=EPOCHS)
    else:
        train_backprop_network_mini_batch(network, data, BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS)
    return time.perf_counter() - start


def measure(config: str, backend: str) -> dict:
    limit = None if config.startswith("dense") else CONV_TRAIN_LIMIT
    train_data = load_mnist_dataset(bss.TRAIN_PATH, limit=limit)
    result = {"epoch": _train_epoch(config, _network(config, backend), train_data)}

    if AFTER:
        network = _network(config, backend)
        start = time.perf_counter()
        network.prepare_dataset(train_data)
        result["prepare"] = time.perf_counter() - start
        result["epoch, loader"] = _train_epoch(config, network, prepared_mnist(bss.TRAIN_PATH, backend, limit=limit))
    return result


def _run_worker(config: str, backend: str) -> dict:
    return run_json_worker([sys.executable, __file__, "worker", config, backend, "--epochs", str(EPOCHS)])


def time_all(configs: list[str], repeats: int) -> dict:
    print(f"trainers: {train.__file__}; prepared path: {AFTER}; epochs per run: {EPOCHS}\n")
    cells = [(config, backend) for config in configs for backend in BACKENDS]
    runs = interleaved_runs(cells, repeats, lambda cell: _run_worker(*cell))

    print(f"median of {repeats}, seconds (one process per measurement)\n")
    print("| config | backend | epoch | prepare | epoch, loader |")
    print("|---|---|---|---|---|")
    for (config, backend), cell_runs in runs.items():
        cells_text = [
            f"{statistics.median(run[m] for run in cell_runs):.3f}" if m in cell_runs[0] else "-" for m in MEASURES
        ]
        print(f"| {config} | {backend} | " + " | ".join(cells_text) + " |")
    return {f"{config} / {backend}": cell_runs for (config, backend), cell_runs in runs.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["time", "worker"])
    parser.add_argument("args", nargs="*", help="worker only: config backend")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1, help="epochs per training run (default 1)")
    parser.add_argument("--out", help="write every run's raw measurements here as JSON")
    args = parser.parse_args(argv)
    global EPOCHS
    EPOCHS = args.epochs

    if args.mode == "worker":
        print(json.dumps(measure(args.args[0], args.args[1])))
    else:
        runs = time_all(args.configs or CONFIGS, args.repeats)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(runs, f, indent=1)


if __name__ == "__main__":
    main()
