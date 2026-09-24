"""
The bit-identical gate for structural refactoring (docs/refactoring.md): trains every array
network, numpy and Rust, from fixed injected weights and records every value it produces, so a
refactoring stage can show that training is unchanged exactly, not within a tolerance.

    python scripts/golden_training_run.py record golden.json   # on main, before the first stage
    python scripts/golden_training_run.py check golden.json    # after each stage

Each network is built with the same injected weights (a seeded random.Random, never
randomize(): pa.uniform can't be seeded), then trained with learn, learn_row, learn_batch and
learn_batch_rows in turn, its snapshot recorded after each. At the end it records classify_rows,
classify_row, classify_state and predict_probabilities (predict_probability for single-output
networks) over the whole dataset, and the snapshot and predictions after a save/load round trip.
The ensembles are assembled from injected single-output classifiers and trained through them.

Floats are recorded as float.hex, so check compares bits. check reports the first differing
value of every network that differs. numpy's products go through BLAS, so a golden file is only
valid on the machine that recorded it; record one on main and check against it on the same
machine.

Dropout: numpy's masks come from np.random, seeded before each dropout network. Rust's can't be
seeded, so the Rust dropout network trains with drop_probability=0.0, which still runs the
training-mode path (every mask entry is 1).
"""

import argparse
import json
import os
import random
import sys
import tempfile

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.model.adam_rust_array_multiclass_backprop_classifier_network import (
    AdamRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.adam_vectorized_multiclass_backprop_classifier_network import (
    AdamVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_array_backprop_classifier_network import (
    CrossEntropyArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_rust_array_backprop_classifier_network import (
    CrossEntropyRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_rust_array_multiclass_backprop_classifier_network import (
    CrossEntropyRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_vectorized_multiclass_backprop_classifier_network import (
    CrossEntropyVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.dropout_rust_array_multiclass_backprop_classifier_network import (
    DropoutRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.dropout_vectorized_multiclass_backprop_classifier_network import (
    DropoutVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.ensemble_array_backprop_classifier_network import EnsembleArrayBackpropClassifierNetwork
from indrajala_ml.model.ensemble_rust_array_backprop_classifier_network import (
    EnsembleRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.l2_rust_array_multiclass_backprop_classifier_network import (
    L2RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.l2_vectorized_multiclass_backprop_classifier_network import (
    L2VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.momentum_rust_array_multiclass_backprop_classifier_network import (
    MomentumRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_vectorized_multiclass_backprop_classifier_network import (
    MomentumVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.relu_rust_array_multiclass_backprop_classifier_network import (
    ReLURustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.relu_vectorized_multiclass_backprop_classifier_network import (
    ReLUVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_rust_array_multiclass_backprop_classifier_network import (
    SoftmaxRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_vectorized_multiclass_backprop_classifier_network import (
    SoftmaxVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)

SEED = 0
LEARNING_RATE = 0.1
ROW_COUNT = 12
BATCHES = [[0, 1, 2, 3], [4, 5, 6, 7]]
LAYER_SIZES = [4, 3]
DIMENSION = 5
CLASS_COUNT = 3
CONV_HEIGHT = CONV_WIDTH = 6
CONV_SPECS = [ConvSpec(3, 2), PoolSpec(2)]
CONV_DENSE_LAYER_SIZES = [4]
ENSEMBLE_SIZE = 3

WRAP = {"numpy": np.array, "rust": pa.Array}

# each name starts with its backend; a dense network is (class, hyperparameters after class_count)
DENSE_NETWORKS = {
    "numpy plain": (VectorizedMultiClassBackpropClassifierNetwork, ()),
    "numpy momentum": (MomentumVectorizedMultiClassBackpropClassifierNetwork, (0.9,)),
    "numpy l2": (L2VectorizedMultiClassBackpropClassifierNetwork, (0.01,)),
    "numpy adam": (AdamVectorizedMultiClassBackpropClassifierNetwork, ()),
    "numpy relu": (ReLUVectorizedMultiClassBackpropClassifierNetwork, ()),
    "numpy softmax": (SoftmaxVectorizedMultiClassBackpropClassifierNetwork, ()),
    "numpy dropout": (DropoutVectorizedMultiClassBackpropClassifierNetwork, (0.3,)),
    "numpy cross-entropy": (CrossEntropyVectorizedMultiClassBackpropClassifierNetwork, ()),
    "rust plain": (RustArrayMultiClassBackpropClassifierNetwork, ()),
    "rust momentum": (MomentumRustArrayMultiClassBackpropClassifierNetwork, (0.9,)),
    "rust l2": (L2RustArrayMultiClassBackpropClassifierNetwork, (0.01,)),
    "rust adam": (AdamRustArrayMultiClassBackpropClassifierNetwork, ()),
    "rust relu": (ReLURustArrayMultiClassBackpropClassifierNetwork, ()),
    "rust softmax": (SoftmaxRustArrayMultiClassBackpropClassifierNetwork, ()),
    "rust dropout": (DropoutRustArrayMultiClassBackpropClassifierNetwork, (0.0,)),
    "rust cross-entropy": (CrossEntropyRustArrayMultiClassBackpropClassifierNetwork, ()),
}
CONV_NETWORKS = {
    "numpy conv": ConvVectorizedMultiClassBackpropClassifierNetwork,
    "rust conv": ConvRustArrayMultiClassBackpropClassifierNetwork,
}
SINGLE_OUTPUT_NETWORKS = {
    "numpy single-output": ArrayBackpropClassifierNetwork,
    "numpy single-output cross-entropy": CrossEntropyArrayBackpropClassifierNetwork,
    "rust single-output": RustArrayBackpropClassifierNetwork,
    "rust single-output cross-entropy": CrossEntropyRustArrayBackpropClassifierNetwork,
}
ENSEMBLES = {
    "numpy ensemble": (EnsembleArrayBackpropClassifierNetwork, ArrayBackpropClassifierNetwork),
    "rust ensemble": (EnsembleRustArrayBackpropClassifierNetwork, RustArrayBackpropClassifierNetwork),
}


def _backend(name: str) -> str:
    return name.split()[0]


def _bits(value):
    # nested lists of floats (or a backend array) as float.hex, so equality is bitwise
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_bits(item) for item in value]
    if isinstance(value, float):
        return float.hex(value)
    return value


def _rows(dimension: int, labels: list) -> list:
    rng = random.Random(f"{SEED} rows {dimension}")
    return [(tuple(rng.uniform(0.0, 1.0) for _ in range(dimension)), labels[i % len(labels)]) for i in range(ROW_COUNT)]


def _random_like(rng: random.Random, value):
    if isinstance(value, list):
        return [_random_like(rng, item) for item in value]
    return rng.uniform(-1.0, 1.0)


def _inject(network, backend: str, rng: random.Random) -> None:
    # every layer's W and b drawn in the shape the network's own snapshot has; a pool layer's
    # empty entry stays empty
    snapshot = []
    for entry in network.snapshot():
        if len(entry) == 0:
            snapshot.append(())
            continue
        W, b = entry
        snapshot.append(tuple(WRAP[backend](_random_like(rng, part.tolist())) for part in (W, b)))
    network.restore(snapshot)


def _train(network, rows: list) -> dict:
    prepared = network.prepare_dataset(rows)
    checkpoints = {}
    for state, category in rows[:3]:
        network.learn(LEARNING_RATE, state, category)
    checkpoints["learn"] = _bits(network.snapshot())
    for index in range(3, 6):
        network.learn_row(LEARNING_RATE, prepared, index)
    checkpoints["learn_row"] = _bits(network.snapshot())
    for batch in BATCHES:
        network.learn_batch(LEARNING_RATE, [rows[i] for i in batch])
    checkpoints["learn_batch"] = _bits(network.snapshot())
    for batch in BATCHES:
        network.learn_batch_rows(LEARNING_RATE, prepared, [i + 4 for i in batch])
    checkpoints["learn_batch_rows"] = _bits(network.snapshot())
    return checkpoints, prepared


def _predictions(network, rows: list, prepared, predict: str) -> dict:
    states = [state for state, _category in rows]
    return {
        "classify_rows": _bits(network.classify_rows(prepared)),
        "classify_row": _bits([network.classify_row(prepared, i) for i in range(len(rows))]),
        "classify_state": _bits([network.classify_state(state) for state in states]),
        predict: _bits([getattr(network, predict)(state) for state in states]),
    }


def _round_trip(network, rows: list, predict: str) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.json")
        network.save(path)
        loaded = type(network).load(path)
    states = [state for state, _category in rows]
    return {
        "snapshot": _bits(loaded.snapshot()),
        predict: _bits([getattr(loaded, predict)(state) for state in states]),
    }


def _run_network(name: str, network, rows: list, predict: str) -> dict:
    _inject(network, _backend(name), random.Random(f"{SEED} weights {name}"))
    np.random.seed(SEED)
    checkpoints, prepared = _train(network, rows)
    return {
        "snapshots": checkpoints,
        "predictions": _predictions(network, rows, prepared, predict),
        "loaded": _round_trip(network, rows, predict),
    }


def _run_ensemble(name: str, ensemble_cls, classifier_cls) -> dict:
    rng = random.Random(f"{SEED} weights {name}")
    classifiers = []
    snapshots = {}
    for class_index in range(ENSEMBLE_SIZE):
        classifier = classifier_cls(LAYER_SIZES, DIMENSION)
        _inject(classifier, _backend(name), rng)
        rows = _rows(DIMENSION, [1.0 if i == class_index else 0.0 for i in range(ENSEMBLE_SIZE)])
        snapshots[f"classifier {class_index}"], _prepared = _train(classifier, rows)
        classifiers.append(classifier)
    ensemble = ensemble_cls(classifiers)

    states = [state for state, _category in _rows(DIMENSION, [0])]
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.json")
        ensemble.save(path)
        loaded = ensemble_cls.load(path)
    return {
        "snapshots": snapshots,
        "predictions": {
            "snapshot": _bits(ensemble.snapshot()),
            "classify_state": _bits([ensemble.classify_state(state) for state in states]),
            "predict_probabilities": _bits([ensemble.predict_probabilities(state) for state in states]),
        },
        "loaded": {
            "snapshot": _bits(loaded.snapshot()),
            "predict_probabilities": _bits([loaded.predict_probabilities(state) for state in states]),
        },
    }


def run_all() -> dict:
    results = {}
    multiclass_rows = _rows(DIMENSION, list(range(CLASS_COUNT)))
    for name, (network_cls, hyperparameters) in DENSE_NETWORKS.items():
        network = network_cls(LAYER_SIZES, DIMENSION, CLASS_COUNT, *hyperparameters)
        results[name] = _run_network(name, network, multiclass_rows, "predict_probabilities")

    conv_rows = _rows(CONV_HEIGHT * CONV_WIDTH, list(range(CLASS_COUNT)))
    for name, network_cls in CONV_NETWORKS.items():
        network = network_cls(CONV_HEIGHT, CONV_WIDTH, CONV_SPECS, CONV_DENSE_LAYER_SIZES, CLASS_COUNT)
        results[name] = _run_network(name, network, conv_rows, "predict_probabilities")

    single_output_rows = _rows(DIMENSION, [0.0, 1.0])
    for name, network_cls in SINGLE_OUTPUT_NETWORKS.items():
        results[name] = _run_network(name, network_cls(LAYER_SIZES, DIMENSION), single_output_rows, "predict_probability")

    for name, (ensemble_cls, classifier_cls) in ENSEMBLES.items():
        results[name] = _run_ensemble(name, ensemble_cls, classifier_cls)
    return results


def _first_difference(expected, actual, path: str = "") -> str | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return f"{path}: keys {sorted(expected)} != {sorted(actual)}"
        for key in expected:
            found = _first_difference(expected[key], actual[key], f"{path}/{key}")
            if found:
                return found
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(expected)} != {len(actual)}"
        for index, (e, a) in enumerate(zip(expected, actual)):
            found = _first_difference(e, a, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if expected == actual else f"{path}: expected {expected!r}, got {actual!r}"


def check(golden: dict, results: dict) -> bool:
    identical = True
    for name in sorted(golden.keys() | results.keys()):
        if name not in results or name not in golden:
            print(f"{name}: only in {'the golden run' if name in golden else 'this run'}")
            identical = False
            continue
        difference = _first_difference(golden[name], results[name])
        if difference:
            print(f"{name}: differs at {difference}")
            identical = False
    print("bit-identical" if identical else "NOT bit-identical", f"({len(golden)} networks in the golden run)")
    return identical


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["record", "check"])
    parser.add_argument("path", help="the golden run's JSON file")
    args = parser.parse_args(argv)

    results = run_all()
    if args.mode == "record":
        os.makedirs(os.path.dirname(args.path) or ".", exist_ok=True)
        with open(args.path, "w") as f:
            json.dump(results, f, indent=1)
        print(f"recorded {len(results)} networks to {args.path}")
    else:
        with open(args.path) as f:
            golden = json.load(f)
        sys.exit(0 if check(golden, results) else 1)


if __name__ == "__main__":
    main()
