"""
The bit-identical gate for structural refactoring (README.md, Refactoring): trains the networks of
all three implementations, pure Python, numpy and Rust, from fixed injected weights and records
every value it produces, so a refactoring stage can show that training is unchanged exactly, not
within a tolerance.

    python scripts/golden_training_run.py record golden.json   # on main, before the first stage
    python scripts/golden_training_run.py check golden.json    # after each stage

Each network is built with the same injected weights (a seeded random.Random, independent of
either backend's RNG), then trained with learn, learn_row, learn_batch and
learn_batch_rows in turn, its snapshot recorded after each. At the end it records classify_rows,
classify_row, classify_state and predict_probabilities (predict_probability for single-output
networks) over the whole dataset, and the snapshot and predictions after a save/load round trip.
The ensembles are assembled from injected single-output classifiers and trained through them.

The pure-Python networks have no prepared-dataset paths, so they train with learn and learn_batch
only, over the same rows, and record classify_state and predict_* only. The pure-Python
single-output networks record no round trip: they had no save when the golden file was recorded.
They are slow, so they train at the same small shapes as the rest.

The conv networks run in two configurations: CONV_SPECS (a conv, then a pool), and
STRIDED_CONV_SPECS (a strided conv, then a second conv, no pool), so a conv layer reads a conv
layer's output.

Floats are recorded as float.hex, so check compares bits. check reports the first differing
value of every network that differs. numpy's products go through BLAS, so a golden file is only
valid on the machine that recorded it; record one on main and check against it on the same
machine.

The residual entries (the residual-connections workplan, stage 5) are Sequential networks of
RESIDUAL_SPECS, two residual blocks in a row, in all three implementations, added after the rest:
no earlier entry moved when they were recorded.

The patch-model and layer-norm entries (the layer-norm and attention workplan, stage 5) are
Sequential networks in all three implementations, added after the residual ones: PATCH_SPECS, the
README's patch model over the conv networks' 6x6 rows under Adam, and LAYER_NORM_SPECS, flat layer
norms after a dropout and a ReLU layer and first in a residual body under momentum (on Rust, the
layers before a layer norm take its downstream and a mask op). No earlier entry moved when they
were recorded.

Dropout: every network's own generator is seeded from SEED, so the numpy and Rust dropout
networks train at the same drop_probability and draw the same masks. The dropout entries were
re-recorded when their masks moved from the global streams to the network's generator (the RNG
generators workplan: numpy and Rust in stage 3, pure Python in stage 4).
"""

import argparse
import json
import math
import os
import random
import sys
import tempfile
from collections.abc import Sequence
from typing import Any, cast

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.model.adam_array_backprop_classifier_network import AdamArrayBackpropClassifierNetwork
from indrajala_ml.model.adam_backprop_classifier_network import AdamBackpropClassifierNetwork
from indrajala_ml.model.adam_multiclass_backprop_classifier_network import AdamMultiClassBackpropClassifierNetwork
from indrajala_ml.model.adam_rust_array_backprop_classifier_network import AdamRustArrayBackpropClassifierNetwork
from indrajala_ml.model.adam_rust_array_multiclass_backprop_classifier_network import (
    AdamRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.adam_vectorized_multiclass_backprop_classifier_network import (
    AdamVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.binary_cross_entropy_backprop_classifier_network import (
    BinaryCrossEntropyBackpropClassifierNetwork,
)
from indrajala_ml.model.classifier_protocols import Example
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_array_backprop_classifier_network import (
    CrossEntropyArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_multiclass_backprop_classifier_network import (
    CrossEntropyMultiClassBackpropClassifierNetwork,
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
from indrajala_ml.model.dropout_array_backprop_classifier_network import DropoutArrayBackpropClassifierNetwork
from indrajala_ml.model.dropout_backprop_classifier_network import DropoutBackpropClassifierNetwork
from indrajala_ml.model.dropout_multiclass_backprop_classifier_network import DropoutMultiClassBackpropClassifierNetwork
from indrajala_ml.model.dropout_rust_array_backprop_classifier_network import DropoutRustArrayBackpropClassifierNetwork
from indrajala_ml.model.dropout_rust_array_multiclass_backprop_classifier_network import (
    DropoutRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.dropout_vectorized_multiclass_backprop_classifier_network import (
    DropoutVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.ensemble_array_backprop_classifier_network import EnsembleArrayBackpropClassifierNetwork
from indrajala_ml.model.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork
from indrajala_ml.model.ensemble_rust_array_backprop_classifier_network import (
    EnsembleRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.fan_in_aware_backprop_classifier_network import FanInAwareBackpropClassifierNetwork
from indrajala_ml.model.l2_array_backprop_classifier_network import L2ArrayBackpropClassifierNetwork
from indrajala_ml.model.l2_regularized_backprop_classifier_network import L2RegularizedBackpropClassifierNetwork
from indrajala_ml.model.l2_regularized_multiclass_backprop_classifier_network import (
    L2RegularizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.l2_rust_array_backprop_classifier_network import L2RustArrayBackpropClassifierNetwork
from indrajala_ml.model.l2_rust_array_multiclass_backprop_classifier_network import (
    L2RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.l2_vectorized_multiclass_backprop_classifier_network import (
    L2VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.layer_specs import (
    Attention,
    Dense,
    InputShape,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    TokenMean,
)
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.momentum_array_backprop_classifier_network import MomentumArrayBackpropClassifierNetwork
from indrajala_ml.model.momentum_backprop_classifier_network import MomentumBackpropClassifierNetwork
from indrajala_ml.model.momentum_conv_multiclass_backprop_classifier_network import (
    MomentumConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_conv_rust_array_multiclass_backprop_classifier_network import (
    MomentumConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_conv_vectorized_multiclass_backprop_classifier_network import (
    MomentumConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_multiclass_backprop_classifier_network import (
    MomentumMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_rust_array_backprop_classifier_network import (
    MomentumRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_rust_array_multiclass_backprop_classifier_network import (
    MomentumRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_vectorized_multiclass_backprop_classifier_network import (
    MomentumVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.relu_array_backprop_classifier_network import ReLUArrayBackpropClassifierNetwork
from indrajala_ml.model.relu_backprop_classifier_network import ReLUBackpropClassifierNetwork
from indrajala_ml.model.relu_multiclass_backprop_classifier_network import ReLUMultiClassBackpropClassifierNetwork
from indrajala_ml.model.relu_rust_array_backprop_classifier_network import ReLURustArrayBackpropClassifierNetwork
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
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.sequential_backprop_network import SequentialMultiClassBackpropClassifierNetwork
from indrajala_ml.model.softmax_multiclass_backprop_classifier_network import (
    SoftmaxMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_rust_array_multiclass_backprop_classifier_network import (
    SoftmaxRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_vectorized_multiclass_backprop_classifier_network import (
    SoftmaxVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import Adam, Momentum, UpdateRule
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.pcg64 import default_rng
from indrajala_ml.prepared_dataset import PreparedDataset

# A network is typed Any here: the script drives the network classes, dense, conv, single-output
# and ensemble, of all three implementations, through the methods they share by name. Recorded
# values (nested lists and dicts of float.hex strings and labels) are JSON, typed Any as json's are.

SEED = 0
LEARNING_RATE = 0.1
ROW_COUNT = 12
BATCHES = [[0, 1, 2, 3], [4, 5, 6, 7]]
LAYER_SIZES = [4, 3]
DIMENSION = 5
CLASS_COUNT = 3
CONV_HEIGHT = CONV_WIDTH = 6
CONV_SPECS = [ConvSpec(3, 2), PoolSpec(2)]
STRIDED_CONV_SPECS = [ConvSpec(3, 2, stride=2), ConvSpec(2, 2)]
CONV_DENSE_LAYER_SIZES = [4]
ENSEMBLE_SIZE = 3
# the pure-Python networks' input_bounds: the rows' range
INPUT_BOUNDS = [(0.0, 1.0)] * DIMENSION

WRAP = {"numpy": np.array, "rust": pa.Array}

# each name starts with its implementation, "python", "numpy" or "rust"; a dense array network is
# (class, hyperparameters after class_count)
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
    "rust dropout": (DropoutRustArrayMultiClassBackpropClassifierNetwork, (0.3,)),
    "rust cross-entropy": (CrossEntropyRustArrayMultiClassBackpropClassifierNetwork, ()),
}
# a conv network is (class, hyperparameters after class_count); each runs in both configurations,
# the strided one under its name plus " strided"
CONV_NETWORKS = {
    "numpy conv": (ConvVectorizedMultiClassBackpropClassifierNetwork, ()),
    "rust conv": (ConvRustArrayMultiClassBackpropClassifierNetwork, ()),
    "numpy momentum conv": (MomentumConvVectorizedMultiClassBackpropClassifierNetwork, (0.9,)),
    "rust momentum conv": (MomentumConvRustArrayMultiClassBackpropClassifierNetwork, (0.9,)),
    "python conv": (ConvMultiClassBackpropClassifierNetwork, ()),
    "python momentum conv": (MomentumConvMultiClassBackpropClassifierNetwork, (0.9,)),
}
# an array single-output network is (class, keyword-only hyperparameters)
SINGLE_OUTPUT_NETWORKS: dict[str, tuple[Any, dict[str, float]]] = {
    "numpy single-output": (ArrayBackpropClassifierNetwork, {}),
    "numpy single-output cross-entropy": (CrossEntropyArrayBackpropClassifierNetwork, {}),
    "rust single-output": (RustArrayBackpropClassifierNetwork, {}),
    "rust single-output cross-entropy": (CrossEntropyRustArrayBackpropClassifierNetwork, {}),
    "numpy single-output relu": (ReLUArrayBackpropClassifierNetwork, {}),
    "numpy single-output dropout": (DropoutArrayBackpropClassifierNetwork, {"drop_probability": 0.3}),
    "numpy single-output momentum": (MomentumArrayBackpropClassifierNetwork, {"momentum": 0.9}),
    "numpy single-output adam": (AdamArrayBackpropClassifierNetwork, {}),
    "numpy single-output l2": (L2ArrayBackpropClassifierNetwork, {"l2_lambda": 0.01}),
    "rust single-output relu": (ReLURustArrayBackpropClassifierNetwork, {}),
    "rust single-output dropout": (DropoutRustArrayBackpropClassifierNetwork, {"drop_probability": 0.3}),
    "rust single-output momentum": (MomentumRustArrayBackpropClassifierNetwork, {"momentum": 0.9}),
    "rust single-output adam": (AdamRustArrayBackpropClassifierNetwork, {}),
    "rust single-output l2": (L2RustArrayBackpropClassifierNetwork, {"l2_lambda": 0.01}),
}
ENSEMBLES = {
    "numpy ensemble": (EnsembleArrayBackpropClassifierNetwork, ArrayBackpropClassifierNetwork),
    "rust ensemble": (EnsembleRustArrayBackpropClassifierNetwork, RustArrayBackpropClassifierNetwork),
    "python ensemble": (EnsembleBackpropClassifierNetwork, BackpropClassifierNetwork),
}
# the pure-Python dense networks: (class, hyperparameters after class_count, or after input_bounds
# for a single-output one)
PYTHON_MULTICLASS_NETWORKS = {
    "python multiclass": (MultiClassBackpropClassifierNetwork, ()),
    "python softmax": (SoftmaxMultiClassBackpropClassifierNetwork, ()),
    "python multiclass momentum": (MomentumMultiClassBackpropClassifierNetwork, (0.9,)),
    "python multiclass l2": (L2RegularizedMultiClassBackpropClassifierNetwork, (0.01,)),
    "python multiclass adam": (AdamMultiClassBackpropClassifierNetwork, ()),
    "python multiclass relu": (ReLUMultiClassBackpropClassifierNetwork, ()),
    "python multiclass dropout": (DropoutMultiClassBackpropClassifierNetwork, (0.3,)),
    "python multiclass cross-entropy": (CrossEntropyMultiClassBackpropClassifierNetwork, ()),
}
PYTHON_SINGLE_OUTPUT_NETWORKS = {
    "python single-output": (BackpropClassifierNetwork, ()),
    "python fan-in-aware": (FanInAwareBackpropClassifierNetwork, ()),
    "python momentum": (MomentumBackpropClassifierNetwork, (0.9,)),
    "python l2": (L2RegularizedBackpropClassifierNetwork, (0.01,)),
    "python adam": (AdamBackpropClassifierNetwork, ()),
    "python relu": (ReLUBackpropClassifierNetwork, ()),
    "python dropout": (DropoutBackpropClassifierNetwork, (0.3,)),
    "python cross-entropy": (BinaryCrossEntropyBackpropClassifierNetwork, ()),
}


# two residual blocks in a row, a sigmoid body and a ReLU one, each ending in its affine layer, under
# momentum: the Sequential network of each implementation
RESIDUAL_SPECS: list[LayerSpec] = [
    Dense(4, activation="relu"),
    Residual((Dense(3), Dense(4, activation="linear", bias=True))),
    Residual((Dense(3, activation="relu"), Dense(4, activation="linear", bias=True))),
    Dense(CLASS_COUNT, output=True),
]
RESIDUAL_NETWORKS = ["numpy residual", "rust residual", "python residual"]


# the README's patch model over the 6x6 image: 4 patches of 3x3, embedded to 4, a position, the
# attention and FFN blocks, the mean, a layer norm and the output, under Adam
PATCH_SPECS: list[LayerSpec] = [
    Patches(3),
    Dense(4, activation="linear", bias=True),
    Position(),
    Residual((LayerNorm(), Attention())),
    Residual((LayerNorm(), Dense(5, activation="relu"), Dense(4, activation="linear", bias=True))),
    TokenMean(),
    LayerNorm(),
    Dense(CLASS_COUNT, output=True),
]
PATCH_NETWORKS = ["numpy patch model", "rust patch model", "python patch model"]
# flat layer norms after a dropout layer, first in a residual body and after a ReLU layer, under
# momentum
LAYER_NORM_SPECS: list[LayerSpec] = [
    Dense(4, dropout=0.3),
    LayerNorm(),
    Residual((LayerNorm(), Dense(3, activation="relu"), Dense(4, activation="linear", bias=True))),
    Dense(4, activation="relu"),
    LayerNorm(),
    Dense(CLASS_COUNT, output=True),
]
LAYER_NORM_NETWORKS = ["numpy layer norm", "rust layer norm", "python layer norm"]


def _sequential_network(name: str, input_shape: InputShape, specs: list[LayerSpec], rule: UpdateRule) -> Any:
    if _backend(name) == "python":
        bounds = [(0.0, 1.0)] * math.prod(input_shape)
        return SequentialMultiClassBackpropClassifierNetwork(input_shape, specs, rule, bounds)
    backend = NUMPY if _backend(name) == "numpy" else RUST
    return SequentialArrayNetwork(input_shape, specs, rule, backend=backend)


def _residual_network(name: str) -> Any:
    return _sequential_network(name, (DIMENSION,), RESIDUAL_SPECS, Momentum(0.9))


def _backend(name: str) -> str:
    return name.split()[0]


def _bits(value: Any) -> Any:
    # nested lists of floats (or a backend array) as float.hex, so equality is bitwise
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_bits(item) for item in cast("list[Any] | tuple[Any, ...]", value)]
    if isinstance(value, float):
        return float.hex(value)
    return value


def _rows[L](dimension: int, labels: Sequence[L]) -> list[Example[L]]:
    rng = random.Random(f"{SEED} rows {dimension}")
    return [(tuple(rng.uniform(0.0, 1.0) for _ in range(dimension)), labels[i % len(labels)]) for i in range(ROW_COUNT)]


def _random_like(rng: random.Random, value: Any) -> Any:
    if isinstance(value, list):
        return [_random_like(rng, item) for item in cast("list[Any]", value)]
    if isinstance(value, tuple):
        return tuple(_random_like(rng, item) for item in cast("tuple[Any, ...]", value))
    return rng.uniform(-1.0, 1.0)


def _inject(network: Any, backend: str, rng: random.Random) -> None:
    if backend == "python":
        # per layer, a (weights, bias) per node or kernel; a pool layer's empty list stays empty
        network.restore(_random_like(rng, network.snapshot()))
        return
    # every layer's parameters (W and b; P; gamma and beta; attention's eight) drawn in order, in
    # the shapes the network's own snapshot has; a parameter-free layer's empty entry stays empty
    network.restore(
        [tuple(WRAP[backend](_random_like(rng, part.tolist())) for part in entry) for entry in network.snapshot()]
    )


def _train(network: Any, rows: Sequence[Example[Any]], backend: str) -> tuple[dict[str, Any], PreparedDataset | None]:
    checkpoints: dict[str, Any] = {}
    if backend == "python":
        # no prepared-dataset paths: learn and learn_batch over the rows the array networks'
        # four paths see, in the same order
        for state, category in rows[:6]:
            network.learn(LEARNING_RATE, state, category)
        checkpoints["learn"] = _bits(network.snapshot())
        for batch in BATCHES:
            network.learn_batch(LEARNING_RATE, [rows[i] for i in batch])
        for batch in BATCHES:
            network.learn_batch(LEARNING_RATE, [rows[i + 4] for i in batch])
        checkpoints["learn_batch"] = _bits(network.snapshot())
        return checkpoints, None

    prepared: PreparedDataset = network.prepare_dataset(rows)
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


def _predictions(
    network: Any, rows: Sequence[Example[Any]], prepared: PreparedDataset | None, predict: str
) -> dict[str, Any]:
    states = [state for state, _category in rows]
    if prepared is None:  # a pure-Python network, which has no classify_rows or classify_row
        return {
            "classify_state": _bits([network.classify_state(state) for state in states]),
            predict: _bits([getattr(network, predict)(state) for state in states]),
        }
    return {
        "classify_rows": _bits(network.classify_rows(prepared)),
        "classify_row": _bits([network.classify_row(prepared, i) for i in range(len(rows))]),
        "classify_state": _bits([network.classify_state(state) for state in states]),
        predict: _bits([getattr(network, predict)(state) for state in states]),
    }


def _round_trip(network: Any, rows: Sequence[Example[Any]], predict: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.json")
        network.save(path)
        loaded = network.__class__.load(path)
    states = [state for state, _category in rows]
    return {
        "snapshot": _bits(loaded.snapshot()),
        predict: _bits([getattr(loaded, predict)(state) for state in states]),
    }


def _run_network(name: str, network: Any, rows: Sequence[Example[Any]], predict: str) -> dict[str, Any]:
    _inject(network, _backend(name), random.Random(f"{SEED} weights {name}"))
    network.rng = default_rng(SEED) if _backend(name) == "python" else network.backend.default_rng(SEED)
    checkpoints, prepared = _train(network, rows, _backend(name))
    result = {
        "snapshots": checkpoints,
        "predictions": _predictions(network, rows, prepared, predict),
    }
    # not the pure-Python single-output networks, which had no save when the golden file was
    # recorded (format 2 gave them one)
    if hasattr(network, "save") and name not in PYTHON_SINGLE_OUTPUT_NETWORKS:
        result["loaded"] = _round_trip(network, rows, predict)
    return result


def _run_ensemble(name: str, ensemble_cls: Any, classifier_cls: Any) -> dict[str, Any]:
    rng = random.Random(f"{SEED} weights {name}")
    classifiers: list[Any] = []
    snapshots: dict[str, Any] = {}
    for class_index in range(ENSEMBLE_SIZE):
        # the array classifiers accept input_bounds and ignore it, as ensemble_train.py relies on
        classifier = classifier_cls(LAYER_SIZES, DIMENSION, INPUT_BOUNDS)
        _inject(classifier, _backend(name), rng)
        rows = _rows(DIMENSION, [1.0 if i == class_index else 0.0 for i in range(ENSEMBLE_SIZE)])
        snapshots[f"classifier {class_index}"], _prepared = _train(classifier, rows, _backend(name))
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


def run_all() -> dict[str, Any]:
    results: dict[str, Any] = {}
    multiclass_rows = _rows(DIMENSION, list(range(CLASS_COUNT)))
    for name, (network_cls, hyperparameters) in DENSE_NETWORKS.items():
        network = network_cls(LAYER_SIZES, DIMENSION, CLASS_COUNT, *hyperparameters)
        results[name] = _run_network(name, network, multiclass_rows, "predict_probabilities")

    conv_rows = _rows(CONV_HEIGHT * CONV_WIDTH, list(range(CLASS_COUNT)))
    for conv_name, (network_cls, hyperparameters) in CONV_NETWORKS.items():
        for name, conv_specs in ((conv_name, CONV_SPECS), (f"{conv_name} strided", STRIDED_CONV_SPECS)):
            network = network_cls(
                CONV_HEIGHT, CONV_WIDTH, conv_specs, CONV_DENSE_LAYER_SIZES, CLASS_COUNT, *hyperparameters
            )
            results[name] = _run_network(name, network, conv_rows, "predict_probabilities")

    single_output_rows = _rows(DIMENSION, [0.0, 1.0])
    for name, (network_cls, hyperparameters) in SINGLE_OUTPUT_NETWORKS.items():
        network = network_cls(LAYER_SIZES, DIMENSION, **hyperparameters)
        results[name] = _run_network(name, network, single_output_rows, "predict_probability")

    for name, (network_cls, hyperparameters) in PYTHON_MULTICLASS_NETWORKS.items():
        network = network_cls(LAYER_SIZES, DIMENSION, INPUT_BOUNDS, CLASS_COUNT, *hyperparameters)
        results[name] = _run_network(name, network, multiclass_rows, "predict_probabilities")

    for name, (network_cls, hyperparameters) in PYTHON_SINGLE_OUTPUT_NETWORKS.items():
        network = network_cls(LAYER_SIZES, DIMENSION, INPUT_BOUNDS, *hyperparameters)
        results[name] = _run_network(name, network, single_output_rows, "predict_probability")

    for name, (ensemble_cls, classifier_cls) in ENSEMBLES.items():
        results[name] = _run_ensemble(name, ensemble_cls, classifier_cls)

    for name in RESIDUAL_NETWORKS:
        results[name] = _run_network(name, _residual_network(name), multiclass_rows, "predict_probabilities")

    for name in PATCH_NETWORKS:
        network = _sequential_network(name, (CONV_HEIGHT, CONV_WIDTH, 1), PATCH_SPECS, Adam())
        results[name] = _run_network(name, network, conv_rows, "predict_probabilities")

    for name in LAYER_NORM_NETWORKS:
        network = _sequential_network(name, (DIMENSION,), LAYER_NORM_SPECS, Momentum(0.9))
        results[name] = _run_network(name, network, multiclass_rows, "predict_probabilities")
    return results


def _first_difference(expected: Any, actual: Any, path: str = "") -> str | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_dict, actual_dict = cast("dict[str, Any]", expected), cast("dict[str, Any]", actual)
        if expected_dict.keys() != actual_dict.keys():
            return f"{path}: keys {sorted(expected_dict)} != {sorted(actual_dict)}"
        for key in expected_dict:
            found = _first_difference(expected_dict[key], actual_dict[key], f"{path}/{key}")
            if found:
                return found
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        expected_list, actual_list = cast("list[Any]", expected), cast("list[Any]", actual)
        if len(expected_list) != len(actual_list):
            return f"{path}: length {len(expected_list)} != {len(actual_list)}"
        for index, (e, a) in enumerate(zip(expected_list, actual_list)):
            found = _first_difference(e, a, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if expected == actual else f"{path}: expected {expected!r}, got {actual!r}"


def check(golden: dict[str, Any], results: dict[str, Any]) -> bool:
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
