"""
The prepared-dataset path trains exactly as the tuple
path does: for every numpy and Rust array network class, learn_row gives bit-for-bit the weights
learn gives, step by step, learn_batch_rows those of learn_batch, and classify_row agrees with
classify_state. The batched accuracy pass predicts what classify_row does, row for
row. The classes are enumerated from the two bases, so a new one can't be missed.
"""

import random
from collections.abc import Callable
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.mnist_data import load_mnist_dataset
from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.networks.array_network_base import ArrayNetworkBase
from indrajala_ml.model.networks.numpy.numpy_array_network_base import NumpyArrayNetworkBase
from indrajala_ml.model.networks.rust.dropout_rust_array_multiclass_backprop_classifier_network import (
    DropoutRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.rust.rust_array_network_base import RustArrayNetworkBase
from indrajala_ml.model.protocols.classifier_protocols import Example
from indrajala_ml.model.specs.layer_specs import Dense
from indrajala_ml.model.specs.update_rules import Adam, Momentum
from indrajala_ml.prepared_dataset import CLASSIFY_CHUNK_ROWS, PreparedDataset, prepared_mnist
from indrajala_ml.train import _training_accuracy
from tests.helpers import all_subclasses, model_modules

MNIST_TRAIN = "data/mnist/mnist-train.bin"

SIDE = 6
DIMENSION = SIDE * SIDE
CLASS_COUNT = 3
CONV_SPECS = [ConvSpec(3, 2), PoolSpec(2)]

model_modules()


def _class_name(cls: type[Any]) -> str:
    return cls.__name__


# a network (class) is Any here: the tests drive every array network class through the methods
# its shape has (multiclass, single-output or conv), each class built as CONSTRUCTORS says
_BASES: set[type[Any]] = {NumpyArrayNetworkBase, RustArrayNetworkBase}
NETWORK_CLASSES: list[type[Any]] = sorted(
    {cls for cls in all_subclasses(ArrayNetworkBase) if cls not in _BASES}, key=_class_name
)

# the sequential networks' layers: kinds and rules no preset combines
SEQUENTIAL_MULTICLASS = [
    Dense(5, activation="relu"),
    Dense(4),
    Dense(CLASS_COUNT, output=True, activation="softmax", loss="cross_entropy"),
]
SEQUENTIAL_SINGLE_OUTPUT = [Dense(5, activation="relu"), Dense(1, output=True, loss="cross_entropy")]

# how to build each class; a class missing here fails test_every_class_has_a_constructor.
# Both twins' generators are reseeded before each step (_seed_step), so the dropout classes
# draw the same masks.
CONSTRUCTORS: dict[str, Callable[[type[Any]], Any]] = {
    "VectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT),
    "AdamVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT),
    "ConvVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls(SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT),
    "CrossEntropyVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT),
    "DropoutVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT, 0.3),
    "L2VectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT, 0.01),
    "MomentumVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT, 0.9),
    "MomentumConvVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls(
        SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT, 0.9
    ),
    "AdamConvVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls(SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT),
    "L2ConvVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls(
        SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT, 0.01
    ),
    "ReLUConvVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls(SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT),
    "DropoutConvVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls(
        SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT, 0.3
    ),
    "CrossEntropyConvVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls(
        SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT
    ),
    "SoftmaxConvVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls(
        SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT
    ),
    "ReLUVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT),
    "SoftmaxVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT),
    "ArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION),
    "CrossEntropyArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION),
    "AdamArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION),
    "DropoutArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, drop_probability=0.3),
    "L2ArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, l2_lambda=0.01),
    "MomentumArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, momentum=0.9),
    "ReLUArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION),
    "RustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT),
    "AdamRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT),
    "ConvRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls(SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT),
    "CrossEntropyRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT),
    "DropoutRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT, 0.3),
    "L2RustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT, 0.01),
    "MomentumRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT, 0.9),
    "MomentumConvRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls(
        SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT, 0.9
    ),
    "AdamConvRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls(SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT),
    "L2ConvRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls(
        SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT, 0.01
    ),
    "ReLUConvRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls(SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT),
    "DropoutConvRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls(
        SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT, 0.3
    ),
    "CrossEntropyConvRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls(
        SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT
    ),
    "SoftmaxConvRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls(
        SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT
    ),
    "ReLURustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT),
    "SoftmaxRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, CLASS_COUNT),
    "RustArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION),
    "CrossEntropyRustArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION),
    "AdamRustArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION),
    "DropoutRustArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, drop_probability=0.3),
    "L2RustArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, l2_lambda=0.01),
    "MomentumRustArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION, momentum=0.9),
    "ReLURustArrayBackpropClassifierNetwork": lambda cls: cls([5], DIMENSION),
    "SequentialVectorizedMultiClassBackpropClassifierNetwork": lambda cls: cls(
        (DIMENSION,), SEQUENTIAL_MULTICLASS, Momentum(0.9)
    ),
    "SequentialRustArrayMultiClassBackpropClassifierNetwork": lambda cls: cls(
        (DIMENSION,), SEQUENTIAL_MULTICLASS, Momentum(0.9)
    ),
    "SequentialArrayBackpropClassifierNetwork": lambda cls: cls((DIMENSION,), SEQUENTIAL_SINGLE_OUTPUT, Adam()),
    "SequentialRustArrayBackpropClassifierNetwork": lambda cls: cls((DIMENSION,), SEQUENTIAL_SINGLE_OUTPUT, Adam()),
}


def _is_rust(cls: type[Any]) -> bool:
    return issubclass(cls, RustArrayNetworkBase)


def _is_binary(cls: type[Any]) -> bool:
    return not hasattr(cls, "predict_probabilities")


def _rows(cls: type[Any], count: int = 20) -> list[Example[float]] | list[Example[int]]:
    rng = random.Random(1)
    labels = [float(i % 2) for i in range(count)] if _is_binary(cls) else [i % CLASS_COUNT for i in range(count)]
    return [(tuple(rng.random() for _ in range(DIMENSION)), label) for label in labels]


def _twin_networks(cls: type[Any]) -> tuple[Any, Any]:
    # two networks with identical starting weights, built through snapshot/restore
    first = CONSTRUCTORS[cls.__name__](cls)
    first.randomize()
    second = CONSTRUCTORS[cls.__name__](cls)
    second.restore(first.snapshot())
    return first, second


def _weights(network: Any) -> list[list[Any]]:
    return [[array.tolist() for array in entry] for entry in network.snapshot()]


def _seed_step(network: Any, step: int) -> None:
    # only the dropout classes draw while training, from the network's own generator
    network.rng = network.backend.default_rng(step)


def test_every_class_has_a_constructor():
    assert len(NETWORK_CLASSES) >= 22
    assert {cls.__name__ for cls in NETWORK_CLASSES} == set(CONSTRUCTORS)


@pytest.mark.parametrize("cls", NETWORK_CLASSES, ids=_class_name)
def test_learn_row_matches_learn_exactly_step_by_step(cls: type[Any]):
    via_tuples, via_rows = _twin_networks(cls)
    rows = _rows(cls)
    prepared = via_rows.prepare_dataset(rows)

    for step, index in enumerate([3, 0, 7, 7, 12, 19, 1]):
        state, label = rows[index]
        _seed_step(via_tuples, step)
        via_tuples.learn(0.5, state, label)
        _seed_step(via_rows, step)
        via_rows.learn_row(0.5, prepared, index)
        assert _weights(via_rows) == _weights(via_tuples), f"step {step}"


@pytest.mark.parametrize("cls", NETWORK_CLASSES, ids=_class_name)
def test_learn_batch_rows_matches_learn_batch_exactly_step_by_step(cls: type[Any]):
    via_tuples, via_rows = _twin_networks(cls)
    rows = _rows(cls)
    prepared = via_rows.prepare_dataset(rows)

    for step, indices in enumerate([[5, 0, 8, 13], [19], [5, 5, 2], list(range(20))]):
        _seed_step(via_tuples, step)
        via_tuples.learn_batch(0.5, [rows[i] for i in indices])
        _seed_step(via_rows, step)
        via_rows.learn_batch_rows(0.5, prepared, indices)
        assert _weights(via_rows) == _weights(via_tuples), f"step {step}"


@pytest.mark.parametrize("cls", NETWORK_CLASSES, ids=_class_name)
def test_classify_row_matches_classify_state(cls: type[Any]):
    network, _ = _twin_networks(cls)
    rows = _rows(cls)
    prepared = network.prepare_dataset(rows)
    assert [network.classify_row(prepared, i) for i in range(len(rows))] == [
        network.classify_state(state) for state, _label in rows
    ]


# more than two chunks, the last one partial
CLASSIFY_ROW_COUNT = 2 * CLASSIFY_CHUNK_ROWS + 6


@pytest.mark.parametrize("cls", NETWORK_CLASSES, ids=_class_name)
def test_classify_rows_matches_classify_row(cls: type[Any]):
    network, _ = _twin_networks(cls)
    prepared = network.prepare_dataset(_rows(cls, CLASSIFY_ROW_COUNT))
    predictions = network.classify_rows(prepared)
    assert predictions == [network.classify_row(prepared, i) for i in range(len(prepared))]
    assert {type(p) for p in predictions} == {float if _is_binary(cls) else int}


@pytest.mark.parametrize("cls", NETWORK_CLASSES, ids=_class_name)
def test_the_prepared_accuracy_pass_matches_the_tuple_one(cls: type[Any]):
    network, _ = _twin_networks(cls)
    rows = _rows(cls, CLASSIFY_ROW_COUNT)
    prepared = network.prepare_dataset(rows)
    assert _training_accuracy(network, rows, prepared) == _training_accuracy(network, rows)


def test_classify_rows_runs_numpy_dropout_in_inference_mode():
    # inference draws no mask, so the pass leaves the network's generator where it was
    cls = next(cls for cls in NETWORK_CLASSES if cls.__name__ == "DropoutVectorizedMultiClassBackpropClassifierNetwork")
    network, _ = _twin_networks(cls)
    prepared = network.prepare_dataset(_rows(cls, CLASSIFY_ROW_COUNT))
    before = network.rng.bit_generator.state
    network.classify_rows(prepared)
    assert network.rng.bit_generator.state == before
    assert not network.hidden_layers[0]._was_training


def test_classify_rows_runs_rust_dropout_in_inference_mode():
    # inference draws no mask, so the pass leaves the network's generator where it was
    cls = DropoutRustArrayMultiClassBackpropClassifierNetwork
    network, _ = _twin_networks(cls)
    prepared = network.prepare_dataset(_rows(cls, CLASSIFY_ROW_COUNT))
    before = network.rng.state
    network.classify_rows(prepared)
    assert network.rng.state == before
    assert not network.hidden_layers[0]._was_training


def _rust_multiclass() -> Any:
    cls = next(cls for cls in NETWORK_CLASSES if cls.__name__ == "RustArrayMultiClassBackpropClassifierNetwork")
    return CONSTRUCTORS[cls.__name__](cls)


def test_the_rust_row_argmax_breaks_ties_as_pa_argmax_does():
    nan = float("nan")
    rows = [
        [1.0, 3.0, 3.0],
        [2.0, 2.0, 2.0],
        [-0.0, 0.0, -1.0],
        [0.0, -0.0, 0.0],
        [nan, 1.0, 2.0],
        [1.0, nan, 2.0],
        [0.2, 0.1, nan],
    ]
    expected = [pa.argmax(pa.Array(row)) for row in rows]
    assert _rust_multiclass()._classify_output_batch(pa.Array(rows)) == expected


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_the_batched_binary_threshold_is_strictly_above_one_half(backend: str):
    name = "ArrayBackpropClassifierNetwork" if backend == "numpy" else "RustArrayBackpropClassifierNetwork"
    cls = next(cls for cls in NETWORK_CLASSES if cls.__name__ == name)
    network = CONSTRUCTORS[name](cls)
    column = [[0.5], [np.nextafter(0.5, 1.0)], [np.nextafter(0.5, 0.0)], [0.9]]
    outputs = np.array(column) if backend == "numpy" else pa.Array(column)
    single = [network._classify_output(np.array(row) if backend == "numpy" else pa.Array(row)) for row in column]
    assert network._classify_output_batch(outputs) == single == [0.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize("cls", NETWORK_CLASSES, ids=_class_name)
def test_training_leaves_the_prepared_matrix_unchanged(cls: type[Any]):
    network, _ = _twin_networks(cls)
    rows = _rows(cls)
    prepared = network.prepare_dataset(rows)
    before = prepared.states.tolist()
    for index in range(len(rows)):
        network.learn_row(0.5, prepared, index)
    network.learn_batch_rows(0.5, prepared, list(range(len(rows))))
    assert prepared.states.tolist() == before


@pytest.mark.parametrize("cls", NETWORK_CLASSES, ids=_class_name)
def test_a_network_rejects_the_other_backends_dataset(cls: type[Any]):
    network, _ = _twin_networks(cls)
    other = PreparedDataset.from_rows(_rows(cls), "numpy" if _is_rust(cls) else "rust")
    with pytest.raises(AssertionError):
        network.learn_row(0.5, other, 0)
    with pytest.raises(AssertionError):
        network.learn_batch_rows(0.5, other, [0, 1])
    with pytest.raises(AssertionError):
        network.classify_row(other, 0)


def test_the_row_paths_train_in_training_mode():
    # the flag the Rust dropout class's layers record, checked directly
    cls = next(cls for cls in NETWORK_CLASSES if cls.__name__ == "DropoutRustArrayMultiClassBackpropClassifierNetwork")
    network, _ = _twin_networks(cls)
    prepared = network.prepare_dataset(_rows(cls))
    hidden = network.layers[0]

    network.learn_row(0.5, prepared, 0)
    assert hidden._was_training
    network.classify_row(prepared, 0)
    assert not hidden._was_training
    network.learn_batch_rows(0.5, prepared, [0, 1])
    assert hidden._was_training


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_from_rows_holds_the_rows_and_labels(backend: str):
    rows = [((0.1, 0.2), 1), ((0.3, 0.4), 0), ((0.5, 0.6), 2)]
    prepared = PreparedDataset.from_rows(rows, backend)
    assert len(prepared) == 3
    assert prepared.backend == backend
    assert prepared.states.tolist() == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    assert prepared.labels == [1, 0, 2]
    assert isinstance(prepared.states, np.ndarray if backend == "numpy" else pa.Array)


def test_a_prepared_dataset_rejects_bad_input():
    with pytest.raises(AssertionError):
        PreparedDataset.from_rows([], "numpy")
    with pytest.raises(AssertionError):
        PreparedDataset.from_rows([((0.1,), 0)], "torch")
    with pytest.raises(AssertionError):
        PreparedDataset(np.zeros((2, 3)), [0], "numpy")


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_prepared_mnist_matches_preparing_the_loaded_tuples(backend: str):
    expected = PreparedDataset.from_rows(load_mnist_dataset(MNIST_TRAIN, limit=50), backend)
    actual = prepared_mnist(MNIST_TRAIN, backend, limit=50)
    assert actual.backend == backend
    assert actual.labels == expected.labels
    assert actual.states.tolist() == expected.states.tolist()
