"""
The prepared dataset's A/B: one training epoch through the
trainers the demos use, before and after the dataset became one backend array (#373, #374).

    python scripts/prepared_dataset_timing.py time [--repeats 5] [--configs ...] [--epochs 1] [--out runs.json]

Every (config, backend, repeat) runs in its own process, never two backends in one, rotating the
order each repeat, and the medians are reported. The script only uses interfaces that exist on
both sides of the change, so old against new runs this same file on each side: scripts/ab.py
(`--bench prepared_dataset_timing`, docs/measurement.md) runs it with each side's worktree on
PYTHONPATH (the output starts with the trainers module it imported):

    python scripts/ab.py run --bench prepared_dataset_timing [--old main] [--new HEAD]

--epochs trains each run for more epochs (one accuracy pass per epoch, plus one before), as a
longer run does; the batched accuracy pass's A/B used it. Configs, each one epoch by default,
from numpy-drawn seed-0 weights, the shuffle seeded 0 (seeded_weights.seeded_shuffle):
- dense B=32 / dense single: 784 -> 30 -> 10 on full MNIST (60000 rows), learning rate 0.5;
- conv B=32 / conv single: the conv demo's "conv" network (ConvSpec(3, 8), dense 32) on its
  2000-row MNIST subset, learning rate 0.5;
- dense dropout B=32 (not run by default; --configs): dense B=32's network and weights with
  dropout at 0.5 on the hidden layer, its masks drawn from the network's generator seeded 0, or
  on a tree from before the network owned one, from the seeded globals. Added for the RNG
  generators workplan's stage 3, which moved the masks onto the network's generator.

Measures (seconds):
- epoch: the trainer given the tuple list, as every demo calls it, including its two
  training-set accuracy passes (before and after the epoch, for the pocket snapshot); after the
  change this includes preparing the dataset once;
- prepare (after only): student.prepare_dataset of the tuple list alone;
- epoch, loader (after only): the trainer given prepared_mnist's dataset, which never builds
  Python floats; its load time is not included.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import TYPE_CHECKING, Any

import indrajala_math_rust as pa
import numpy as np
from process_runs import interleaved_runs, run_json_worker
from seeded_weights import seeded_randomized, seeded_shuffle

from indrajala_ml import batch_size_scaling as bss
from indrajala_ml import train
from indrajala_ml.mnist_data import load_mnist_dataset
from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.networks.numpy.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.numpy.dropout_vectorized_multiclass_backprop_classifier_network import (
    DropoutVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.rust.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.rust.dropout_rust_array_multiclass_backprop_classifier_network import (
    DropoutRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.train import train_backprop_network_mini_batch, train_linear_classifier_network

# types only: with an old checkout first on PYTHONPATH (see AFTER) the script imports nothing new
if TYPE_CHECKING:
    from indrajala_ml.model.networks.numpy.vectorized_multiclass_backprop_classifier_network import (
        VectorizedMultiClassBackpropClassifierNetwork,
    )
    from indrajala_ml.model.networks.rust.rust_array_multiclass_backprop_classifier_network import (
        RustArrayMultiClassBackpropClassifierNetwork,
    )
    from indrajala_ml.model.protocols.classifier_protocols import Example
    from indrajala_ml.prepared_dataset import PreparedDataset

    # the dense and conv networks of either backend
    Network = VectorizedMultiClassBackpropClassifierNetwork | RustArrayMultiClassBackpropClassifierNetwork

# indrajala_ml is a namespace package, so with an old checkout first on PYTHONPATH the new
# prepared_dataset module can still be imported from this one; the trainers themselves tell
# which side of the change is running
AFTER = hasattr(train, "_prepared_for")

BACKENDS = ["numpy", "rust"]
CONFIGS = ["dense B=32", "dense single", "conv B=32", "conv single"]
# the configs --configs can name: the default ones, then the opt-in ones
ALL_CONFIGS = [*CONFIGS, "dense dropout B=32"]
DROP_PROBABILITY = 0.5
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
DROPOUT_CLASSES = {
    "numpy": DropoutVectorizedMultiClassBackpropClassifierNetwork,
    "rust": DropoutRustArrayMultiClassBackpropClassifierNetwork,
}


def _dropout_network(backend: str) -> Any:
    network = DROPOUT_CLASSES[backend](bss.LAYER_SIZES, bss.DIMENSION, bss.CLASS_COUNT, DROP_PROBABILITY)
    network.restore(bss.initial_network(backend, 0.0, SEED).snapshot())
    if hasattr(network, "rng"):
        network.rng = network.backend.default_rng(SEED)
    else:  # a tree whose dropout layers draw from the globals
        np.random.seed(SEED)
        pa.seed(SEED)
    return network


def _network(config: str, backend: str) -> Network:
    if config == "dense dropout B=32":
        return _dropout_network(backend)
    if config.startswith("dense"):
        return bss.initial_network(backend, 0.0, SEED)
    snapshot = seeded_randomized(
        ConvVectorizedMultiClassBackpropClassifierNetwork, SEED, 28, 28, CONV_SPECS, CONV_DENSE_LAYER_SIZES, 10
    ).snapshot()
    network = CONV_CLASSES[backend](28, 28, CONV_SPECS, CONV_DENSE_LAYER_SIZES, 10)
    network.restore(snapshot)
    return network


def _train_epoch(config: str, network: Network, data: list[Example[int]] | PreparedDataset) -> float:
    trainer = train_linear_classifier_network if config.endswith("single") else train_backprop_network_mini_batch
    shuffle = seeded_shuffle(trainer, SEED)
    start = time.perf_counter()
    if config.endswith("single"):
        train_linear_classifier_network(network, data, learning_rate=LEARNING_RATE, epochs=EPOCHS, **shuffle)
    else:
        train_backprop_network_mini_batch(
            network, data, BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS, **shuffle
        )
    return time.perf_counter() - start


def measure(config: str, backend: str) -> dict[str, float]:
    limit = None if config.startswith("dense") else CONV_TRAIN_LIMIT
    train_data = load_mnist_dataset(bss.TRAIN_PATH, limit=limit)
    result = {"epoch": _train_epoch(config, _network(config, backend), train_data)}

    if AFTER:
        from indrajala_ml.prepared_dataset import prepared_mnist

        network = _network(config, backend)
        start = time.perf_counter()
        network.prepare_dataset(train_data)
        result["prepare"] = time.perf_counter() - start
        result["epoch, loader"] = _train_epoch(config, network, prepared_mnist(bss.TRAIN_PATH, backend, limit=limit))
    return result


def _run_worker(config: str, backend: str) -> dict[str, Any]:
    return run_json_worker([sys.executable, __file__, "worker", config, backend, "--epochs", str(EPOCHS)])


def time_all(configs: list[str], repeats: int) -> dict[str, list[dict[str, Any]]]:
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
    parser.add_argument("--configs", nargs="+", choices=ALL_CONFIGS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1, help="epochs per training run (default 1)")
    parser.add_argument("--out", help="write every run's raw measurements here as JSON")
    args = parser.parse_args(argv)
    global EPOCHS
    EPOCHS = args.epochs  # pyright: ignore[reportConstantRedefinition]  (--epochs sets it, once)

    if args.mode == "worker":
        print(json.dumps(measure(args.args[0], args.args[1])))
    else:
        runs = time_all(args.configs or CONFIGS, args.repeats)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(runs, f, indent=1)


if __name__ == "__main__":
    main()
