import random
from typing import Any, cast

import numpy as np
import pytest

from indrajala_ml import batch_size_scaling as bss
from indrajala_ml.mnist_data import load_mnist_dataset
from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.classifier_protocols import Example
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.layer_specs import BatchNorm, Dense
from indrajala_ml.model.momentum_conv_rust_array_multiclass_backprop_classifier_network import (
    MomentumConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_conv_vectorized_multiclass_backprop_classifier_network import (
    MomentumConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import SGD, Momentum
from indrajala_ml.train import train_backprop_network_mini_batch


@pytest.fixture(scope="module")
def mnist_subset() -> tuple[list[Example[int]], list[Example[int]]]:
    return load_mnist_dataset(bss.TRAIN_PATH, limit=200), load_mnist_dataset(bss.TEST_PATH, limit=100)


def test_scaled_learning_rate_is_linear_in_batch_size():
    assert bss.scaled_learning_rate(3.0, 32) == 3.0
    assert bss.scaled_learning_rate(3.0, 128) == 12.0
    assert bss.scaled_learning_rate(0.5, 1024) == 16.0


def test_warmup_steps_is_warmup_epochs_in_batches_rounded_up():
    assert bss.warmup_steps(0.0, 60000, 32) == 0
    assert bss.warmup_steps(0.25, 60000, 32) == 469  # 468.75
    assert bss.warmup_steps(1.0, 60000, 1024) == 59  # 58.59
    assert bss.warmup_steps(0.25, 60000, 1024) == 15  # 14.65


def test_learning_rate_schedule_is_constant_without_warmup_and_ramps_with_it():
    assert bss.learning_rate_schedule(2.0, 0) == 2.0
    schedule = bss.learning_rate_schedule(2.0, 4)
    assert callable(schedule)
    assert [schedule(step) for step in range(6)] == [0.5, 1.0, 1.5, 2.0, 2.0, 2.0]


def _numpy(array: object) -> FloatArray:
    # a numpy network's snapshot entry (bss.initial_network's type spans both backends)
    assert isinstance(array, np.ndarray)
    return cast(FloatArray, array)


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_initial_network_is_identical_across_backends(momentum: float):
    numpy_weights = bss.initial_network("numpy", momentum, seed=7).snapshot()
    rust_weights = bss.initial_network("rust", momentum, seed=7).snapshot()
    for (numpy_W, numpy_b), (rust_W, rust_b) in zip(numpy_weights, rust_weights):
        assert np.array_equal(_numpy(numpy_W), np.array(rust_W.tolist()))
        assert np.array_equal(_numpy(numpy_b), np.array(rust_b.tolist()))


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_initial_conv_network_is_identical_across_backends(momentum: float):
    numpy_weights = bss.initial_network("numpy", momentum, seed=7, architecture="conv").snapshot()
    rust_weights = bss.initial_network("rust", momentum, seed=7, architecture="conv").snapshot()
    assert len(numpy_weights) == 3  # the conv layer, the dense 32, the output layer
    for (numpy_W, numpy_b), (rust_W, rust_b) in zip(numpy_weights, rust_weights):
        assert np.array_equal(_numpy(numpy_W), np.array(rust_W.tolist()))
        assert np.array_equal(_numpy(numpy_b), np.array(rust_b.tolist()))


@pytest.mark.parametrize(
    "backend, momentum, network_cls",
    [
        ("numpy", 0.0, ConvVectorizedMultiClassBackpropClassifierNetwork),
        ("numpy", 0.9, MomentumConvVectorizedMultiClassBackpropClassifierNetwork),
        ("rust", 0.0, ConvRustArrayMultiClassBackpropClassifierNetwork),
        ("rust", 0.9, MomentumConvRustArrayMultiClassBackpropClassifierNetwork),
    ],
)
def test_initial_conv_network_has_momentum_only_above_zero(backend: str, momentum: float, network_cls: type):
    network = bss.initial_network(backend, momentum, seed=0, architecture="conv")
    assert type(network) is network_cls
    assert getattr(network, "momentum", 0.0) == momentum


def test_initial_network_rejects_unknown_architectures():
    with pytest.raises(ValueError, match="unknown architecture"):
        bss.initial_network("rust", 0.0, seed=0, architecture="lstm")


def test_initial_network_differs_between_seeds():
    first = bss.initial_network("numpy", 0.0, seed=0).snapshot()
    second = bss.initial_network("numpy", 0.0, seed=1).snapshot()
    assert not np.array_equal(_numpy(first[0][0]), _numpy(second[0][0]))


def test_train_epoch_takes_the_same_steps_as_the_trainer(mnist_subset: tuple[list[Example[int]], list[Example[int]]]):
    # the study's loop is the trainer's minus the pocket snapshot: from the same weights and
    # shuffle seed, one improving epoch leaves both networks with identical weights (the pocket
    # keeps the last epoch when it is the best)
    train_data, _test_data = mnist_subset
    schedule = bss.learning_rate_schedule(2.0, 3)

    trainer_network = bss.initial_network("numpy", 0.9, seed=2)  # a seed whose one epoch improves
    random.seed(11)
    result = train_backprop_network_mini_batch(trainer_network, train_data, 32, learning_rate=schedule, epochs=1)
    assert result.diagnostic.best_epoch_index == 0

    study_network = bss.initial_network("numpy", 0.9, seed=2)
    random.seed(11)
    steps, step_seconds = bss.train_epoch(study_network, train_data, 32, schedule, first_step=0)

    assert steps == 7  # ceil(200 / 32)
    assert step_seconds > 0.0
    for (trainer_W, trainer_b), (study_W, study_b) in zip(trainer_network.snapshot(), study_network.snapshot()):
        assert np.array_equal(_numpy(trainer_W), _numpy(study_W))
        assert np.array_equal(_numpy(trainer_b), _numpy(study_b))


def test_train_and_evaluate_reports_every_epoch(mnist_subset: tuple[list[Example[int]], list[Example[int]]]):
    train_data, test_data = mnist_subset
    result = bss.train_and_evaluate("rust", train_data, test_data, 64, 2.0, 1.0, 0.0, epochs=3, seed=0)
    assert len(result["test_accuracies"]) == 3
    assert all(0.0 <= value <= 1.0 for value in result["test_accuracies"])
    assert result["steps"] == 3 * 4  # ceil(200 / 64) per epoch
    assert len(result["step_seconds"]) == 3


def test_train_and_evaluate_is_reproducible_for_a_seed(mnist_subset: tuple[list[Example[int]], list[Example[int]]]):
    train_data, test_data = mnist_subset
    first = bss.train_and_evaluate("rust", train_data, test_data, 32, 2.0, 0.25, 0.9, epochs=2, seed=5)
    second = bss.train_and_evaluate("rust", train_data, test_data, 32, 2.0, 0.25, 0.9, epochs=2, seed=5)
    assert first["test_accuracies"] == second["test_accuracies"]


def test_train_and_evaluate_trains_the_conv_network(mnist_subset: tuple[list[Example[int]], list[Example[int]]]):
    train_data, test_data = mnist_subset
    first = bss.train_and_evaluate(
        "rust", train_data, test_data, 64, 0.5, 1.0, 0.0, epochs=2, seed=0, architecture="conv"
    )
    second = bss.train_and_evaluate(
        "rust", train_data, test_data, 64, 0.5, 1.0, 0.0, epochs=2, seed=0, architecture="conv"
    )
    assert first["steps"] == 2 * 4
    assert first["test_accuracies"] == second["test_accuracies"]


def test_train_and_evaluate_trains_the_momentum_conv_network(
    mnist_subset: tuple[list[Example[int]], list[Example[int]]],
):
    # reproducible for a seed, and momentum changes the run: from the same weights and shuffle,
    # momentum 0.9 ends somewhere other than 0.0
    train_data, test_data = mnist_subset
    runs = [
        bss.train_and_evaluate(
            "rust", train_data, test_data, 64, 0.25, 1.0, momentum, epochs=2, seed=0, architecture="conv"
        )
        for momentum in (0.9, 0.9, 0.0)
    ]
    assert runs[0]["steps"] == 2 * 4
    assert runs[0]["test_accuracies"] == runs[1]["test_accuracies"]
    assert runs[0]["test_accuracies"] != runs[2]["test_accuracies"]


def _arrays(snapshot: list[tuple[Any, ...]]) -> list[list[Any]]:
    # a snapshot of either backend (numpy arrays or pa.Arrays) as nested lists, entry by entry
    return [[array.tolist() for array in entry] for entry in snapshot]


@pytest.mark.parametrize("group_size", [None, 32])
def test_conv_batch_norm_specs_put_batch_norm_on_every_hidden_layer(group_size: int | None):
    assert bss.conv_batch_norm_specs(group_size) == [
        ConvSpec(3, 8, activation="linear"),
        BatchNorm("relu", group_size=group_size),
        Dense(32, activation="linear"),
        BatchNorm("sigmoid", group_size=group_size),
        Dense(10, output=True),
    ]


@pytest.mark.parametrize("group_size", [None, 32])
@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_initial_conv_batch_norm_network_is_identical_across_backends(momentum: float, group_size: int | None):
    numpy_network: Any = bss.initial_network("numpy", momentum, 7, "conv-bn", group_size)
    rust_network: Any = bss.initial_network("rust", momentum, 7, "conv-bn", group_size)
    numpy_weights = numpy_network.snapshot()
    # linear conv (W,), batch norm (gamma, beta, running mean, running var), linear dense (W,), batch
    # norm, the output layer (W, b)
    assert [len(entry) for entry in numpy_weights] == [1, 4, 1, 4, 2]
    assert _arrays(numpy_weights) == _arrays(rust_network.snapshot())
    assert numpy_network.update_rule == rust_network.update_rule == (Momentum(momentum) if momentum else SGD())


def test_initial_conv_batch_norm_weights_depend_on_the_seed_only():
    reference = _arrays(bss.initial_network("rust", 0.0, 3, "conv-bn").snapshot())
    for momentum, group_size in [(0.9, None), (0.0, 32), (0.9, 32)]:
        assert _arrays(bss.initial_network("rust", momentum, 3, "conv-bn", group_size).snapshot()) == reference
    assert _arrays(bss.initial_network("rust", 0.0, 4, "conv-bn").snapshot()) != reference


def test_initial_network_takes_a_group_size_for_conv_bn_only():
    with pytest.raises(AssertionError, match="conv-bn only"):
        bss.initial_network("rust", 0.0, 0, "conv", 32)


def test_conv_batch_norm_arms_are_the_same_bits_at_batch_32(
    mnist_subset: tuple[list[Example[int]], list[Example[int]]],
):
    # a ghost group as large as the batch is plain batch norm (D3), so the two arms share the
    # batch-32 band; the final batch of 8 (200 rows) is one group in both
    train_data, test_data = mnist_subset
    plain = bss.train_and_evaluate("rust", train_data, test_data, 32, 2.0, 1.0, 0.9, 2, 0, "conv-bn", None)
    ghost = bss.train_and_evaluate("rust", train_data, test_data, 32, 2.0, 1.0, 0.9, 2, 0, "conv-bn", 32)
    assert plain["test_accuracies"] == ghost["test_accuracies"]


@pytest.mark.parametrize("batch_size", [128, 512])
def test_conv_batch_norm_trains_with_the_full_mnist_final_batch(batch_size: int):
    # 60000 rows leave a final batch of 96 at B = 128 and 512, three ghost groups of 32: no batch
    # or group of one is refused. Ghost groups change the run against plain batch norm.
    train_data = load_mnist_dataset(bss.TRAIN_PATH, limit=batch_size + 96)
    test_data = load_mnist_dataset(bss.TEST_PATH, limit=100)
    runs = {
        group_size: bss.train_and_evaluate(
            "rust", train_data, test_data, batch_size, 2.0, 0.0, 0.0, 1, 0, "conv-bn", group_size
        )
        for group_size in (None, 32)
    }
    assert runs[None]["steps"] == runs[32]["steps"] == 2
    assert runs[None]["test_accuracies"] != runs[32]["test_accuracies"]
