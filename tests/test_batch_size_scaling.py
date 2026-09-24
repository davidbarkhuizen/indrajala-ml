import random

import numpy as np
import pytest

from indrajala_ml import batch_size_scaling as bss
from indrajala_ml.mnist_data import load_mnist_dataset
from indrajala_ml.train import train_backprop_network_mini_batch


@pytest.fixture(scope="module")
def mnist_subset():
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
    assert [schedule(step) for step in range(6)] == [0.5, 1.0, 1.5, 2.0, 2.0, 2.0]


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_initial_network_is_identical_across_backends(momentum):
    numpy_weights = bss.initial_network("numpy", momentum, seed=7).snapshot()
    rust_weights = bss.initial_network("rust", momentum, seed=7).snapshot()
    for (numpy_W, numpy_b), (rust_W, rust_b) in zip(numpy_weights, rust_weights):
        assert np.array_equal(numpy_W, np.array(rust_W.tolist()))
        assert np.array_equal(numpy_b, np.array(rust_b.tolist()))


def test_initial_network_differs_between_seeds():
    first = bss.initial_network("numpy", 0.0, seed=0).snapshot()
    second = bss.initial_network("numpy", 0.0, seed=1).snapshot()
    assert not np.array_equal(first[0][0], second[0][0])


def test_train_epoch_takes_the_same_steps_as_the_trainer(mnist_subset):
    # the study's loop is the trainer's minus the pocket snapshot: from the same weights and
    # shuffle seed, one improving epoch leaves both networks with identical weights (the pocket
    # keeps the last epoch when it is the best)
    train_data, _test_data = mnist_subset
    schedule = bss.learning_rate_schedule(2.0, 3)

    trainer_network = bss.initial_network("numpy", 0.9, seed=3)
    random.seed(11)
    result = train_backprop_network_mini_batch(trainer_network, train_data, 32, learning_rate=schedule, epochs=1)
    assert result.diagnostic.best_epoch_index == 0

    study_network = bss.initial_network("numpy", 0.9, seed=3)
    random.seed(11)
    steps, step_seconds = bss.train_epoch(study_network, train_data, 32, schedule, first_step=0)

    assert steps == 7  # ceil(200 / 32)
    assert step_seconds > 0.0
    for (trainer_W, trainer_b), (study_W, study_b) in zip(trainer_network.snapshot(), study_network.snapshot()):
        assert np.array_equal(trainer_W, study_W)
        assert np.array_equal(trainer_b, study_b)


def test_train_and_evaluate_reports_every_epoch(mnist_subset):
    train_data, test_data = mnist_subset
    result = bss.train_and_evaluate("rust", train_data, test_data, 64, 2.0, 1.0, 0.0, epochs=3, seed=0)
    assert len(result["test_accuracies"]) == 3
    assert all(0.0 <= value <= 1.0 for value in result["test_accuracies"])
    assert result["steps"] == 3 * 4  # ceil(200 / 64) per epoch
    assert len(result["step_seconds"]) == 3


def test_train_and_evaluate_is_reproducible_for_a_seed(mnist_subset):
    train_data, test_data = mnist_subset
    first = bss.train_and_evaluate("rust", train_data, test_data, 32, 2.0, 0.25, 0.9, epochs=2, seed=5)
    second = bss.train_and_evaluate("rust", train_data, test_data, 32, 2.0, 0.25, 0.9, epochs=2, seed=5)
    assert first["test_accuracies"] == second["test_accuracies"]
