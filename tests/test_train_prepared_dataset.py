"""
The trainers' prepared-dataset path (docs/optimizations/implemented.md): an array network
student trains from one backend matrix, prepared once per run or passed in by the caller,
visiting exactly the examples, in exactly the order, the tuple path does for the same seed.
The existing seeded end-to-end pins (the conv and multiclass pipelines) now run through this
path unchanged.
"""

import random
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from indrajala_ml.mnist_data import load_mnist_dataset
from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.classifier_protocols import Example, State
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.prepared_dataset import PreparedDataset, prepared_mnist
from indrajala_ml.train import train_backprop_network_mini_batch, train_linear_classifier_network

MNIST_TRAIN = "data/mnist/mnist-train.bin"


class _TupleRecorder:
    """A student without prepare_dataset: the trainers give it tuples. Records each state seen."""

    def __init__(self) -> None:
        self.learned: list[Example[int]] = []
        self.batches: list[list[Example[int]]] = []

    def learn(self, learning_rate: float, state: State, category: int) -> None:
        self.learned.append((state, category))

    def learn_batch(self, learning_rate: float, batch: Sequence[Example[int]]) -> None:
        self.batches.append(list(batch))

    def classify_state(self, state: State) -> int:
        return 0

    def snapshot(self) -> None:
        return None

    def restore(self, snapshot: object) -> None:
        pass


class _RowRecorder(_TupleRecorder):
    """A student with the prepared-path methods: records the rows it's given, as tuples again."""

    def prepare_dataset(self, rows: Sequence[Example[int]]) -> PreparedDataset:
        return PreparedDataset.from_rows(rows, "numpy")

    def learn(self, learning_rate: float, state: State, category: int) -> None:
        raise AssertionError("an array student must be trained through learn_row")

    def learn_batch(self, learning_rate: float, batch: Sequence[Example[int]]) -> None:
        raise AssertionError("an array student must be trained through learn_batch_rows")

    def classify_state(self, state: State) -> int:
        raise AssertionError("an array student's accuracy passes must use classify_rows")

    def _row(self, prepared: PreparedDataset, index: int) -> Example[int]:
        states = cast(FloatArray, prepared.states)  # prepare_dataset's, numpy
        return tuple(states[index].tolist()), prepared.labels[index]

    def learn_row(self, learning_rate: float, prepared: PreparedDataset, index: int) -> None:
        self.learned.append(self._row(prepared, index))

    def learn_batch_rows(self, learning_rate: float, prepared: PreparedDataset, indices: Sequence[int]) -> None:
        self.batches.append([self._row(prepared, index) for index in indices])

    def classify_row(self, prepared: PreparedDataset, index: int) -> int:
        raise AssertionError("an array student's accuracy passes must use classify_rows")

    def classify_rows(self, prepared: PreparedDataset) -> list[int]:
        return [0] * len(prepared)


def _rows(count: int = 23) -> list[Example[int]]:
    rng = random.Random(4)
    return [(tuple(rng.random() for _ in range(3)), i % 3) for i in range(count)]


def test_single_example_visits_the_same_examples_in_the_same_order():
    rows = _rows()
    via_tuples, via_rows = _TupleRecorder(), _RowRecorder()
    train_linear_classifier_network(via_tuples, rows, epochs=3)
    train_linear_classifier_network(via_rows, rows, epochs=3)
    assert via_rows.learned == via_tuples.learned == rows * 3


@pytest.mark.parametrize("reshuffle_each_epoch", [True, False])
def test_mini_batch_makes_the_same_batches_for_the_same_seed(reshuffle_each_epoch: bool):
    rows = _rows()
    via_tuples, via_rows = _TupleRecorder(), _RowRecorder()
    random.seed(11)
    train_backprop_network_mini_batch(via_tuples, rows, 5, epochs=3, reshuffle_each_epoch=reshuffle_each_epoch)
    random.seed(11)
    train_backprop_network_mini_batch(via_rows, rows, 5, epochs=3, reshuffle_each_epoch=reshuffle_each_epoch)
    assert via_rows.batches == via_tuples.batches
    assert len(via_rows.batches) == 3 * 5  # the final undersized batch of 3 is kept
    assert (via_rows.batches[:5] == [rows[i : i + 5] for i in range(0, 23, 5)]) is not reshuffle_each_epoch


def _numpy_network() -> VectorizedMultiClassBackpropClassifierNetwork:
    np.random.seed(3)
    return VectorizedMultiClassBackpropClassifierNetwork.randomized([4], 3, 3)


def _weights(
    network: VectorizedMultiClassBackpropClassifierNetwork | RustArrayMultiClassBackpropClassifierNetwork,
) -> list[list[Any]]:
    return [[array.tolist() for array in entry] for entry in network.snapshot()]


def test_a_caller_prepared_dataset_trains_exactly_as_the_tuple_list():
    rows = _rows()
    via_tuples, via_prepared = _numpy_network(), _numpy_network()

    random.seed(5)
    tuple_result = train_backprop_network_mini_batch(via_tuples, rows, 4, learning_rate=0.5, epochs=4)
    random.seed(5)
    prepared_result = train_backprop_network_mini_batch(
        via_prepared, PreparedDataset.from_rows(rows, "numpy"), 4, learning_rate=0.5, epochs=4
    )
    assert _weights(via_prepared) == _weights(via_tuples)
    assert prepared_result.diagnostic.epoch_training_accuracies == tuple_result.diagnostic.epoch_training_accuracies

    via_tuples, via_prepared = _numpy_network(), _numpy_network()
    train_linear_classifier_network(via_tuples, rows, learning_rate=0.5, epochs=2)
    train_linear_classifier_network(via_prepared, PreparedDataset.from_rows(rows, "numpy"), learning_rate=0.5, epochs=2)
    assert _weights(via_prepared) == _weights(via_tuples)


def test_a_loader_prepared_mnist_dataset_trains_exactly_as_the_loaded_tuples():
    first = RustArrayMultiClassBackpropClassifierNetwork.randomized([8], 784, 10)
    second = RustArrayMultiClassBackpropClassifierNetwork([8], 784, 10)
    second.restore(first.snapshot())

    random.seed(2)
    train_backprop_network_mini_batch(first, load_mnist_dataset(MNIST_TRAIN, limit=64), 16, epochs=2)
    random.seed(2)
    train_backprop_network_mini_batch(second, prepared_mnist(MNIST_TRAIN, "rust", limit=64), 16, epochs=2)
    assert _weights(second) == _weights(first)


def test_a_student_without_the_row_methods_rejects_a_prepared_dataset():
    student = BackpropClassifierNetwork.randomized([2], 3, [(0.0, 1.0)] * 3)
    prepared = PreparedDataset.from_rows([((0.1, 0.2, 0.3), 1.0)], "numpy")
    with pytest.raises(AssertionError):
        train_linear_classifier_network(student, prepared)
    with pytest.raises(AssertionError):
        train_backprop_network_mini_batch(student, prepared, 1)
