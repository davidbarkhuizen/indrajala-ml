import random
from collections.abc import Sequence

import pytest

from indrajala_ml.data.targets import XORTarget
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.networks.python.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.networks.python.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.model.protocols.classifier_protocols import Example, StateClassifier
from indrajala_ml.pcg64 import default_rng
from indrajala_ml.training.train import train_backprop_network_mini_batch, train_linear_classifier_network
from indrajala_ml.training.training_data import random_alternating_training_data, reachable_reference_and_training_data
from tests.helpers import approx


def _training_accuracy(student: StateClassifier[float], training_data: Sequence[Example[float]]) -> float:
    return sum(1 for state, category in training_data if student.classify_state(state) == category) / len(training_data)


def test_train_linear_classifier_network_keeps_the_best_epoch_not_the_last():

    # measured per-epoch accuracy without pocket tracking, from weight seed 0: 0.826, 0.745,
    # 0.870, 0.816, 0.814, 0.850, 0.835, 0.842, 0.808, 0.827, so the best epoch (2) isn't the last,
    # on training data from random.Random(0)

    # XOR isn't representable by an AND/OR/k-of-n gate over cardinality=3 half-planes
    bounds = square_bounds(10.0)
    training_data = random_alternating_training_data(1000, XORTarget(bounds), rng=random.Random(0))

    student = LinearClassifierNetwork.randomized(3, 2, bounds, required_active=2, seed=0)
    result = train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=10)

    assert _training_accuracy(student, training_data) == approx(0.87)

    # best epoch index 2, last 9: a plateau
    diagnostic = result.diagnostic
    assert diagnostic.epoch_training_accuracies == [
        approx(a) for a in [0.826, 0.745, 0.870, 0.816, 0.814, 0.850, 0.835, 0.842, 0.808, 0.827]
    ]
    assert diagnostic.best_epoch_index == 2
    assert diagnostic.best_training_accuracy == approx(0.87)
    assert diagnostic.plateaued is True
    assert diagnostic.converged is False
    assert diagnostic.still_improving is False


def test_train_linear_classifier_network_pocket_tracking_is_a_no_op_when_it_converges():

    # when training does converge, the best epoch and the last epoch coincide, so pocket
    # tracking shouldn't change the well-established convergence behavior at all. The reference
    # and the student draw from different seeds, so the student doesn't start as the reference
    bounds = square_bounds(10.0)
    _reference, training_data = reachable_reference_and_training_data(
        1, 2, bounds, 400, rng=default_rng(1), data_rng=random.Random(1)
    )
    student = LinearClassifierNetwork.randomized(1, 2, bounds, seed=101)

    result = train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=5)

    assert _training_accuracy(student, training_data) >= 0.99

    # hasn't hit exact convergence within 5 epochs yet, but the last epoch is still the best
    # one seen - this is "still improving", not a plateau
    diagnostic = result.diagnostic
    assert diagnostic.best_epoch_index == len(diagnostic.epoch_training_accuracies) - 1
    assert diagnostic.still_improving is True
    assert diagnostic.converged is False
    assert diagnostic.plateaued is False


def test_train_linear_classifier_network_diagnostic_reports_converged():

    bounds = square_bounds(10.0)
    _reference, training_data = reachable_reference_and_training_data(
        1, 2, bounds, 400, rng=default_rng(1), data_rng=random.Random(1)
    )
    student = LinearClassifierNetwork.randomized(1, 2, bounds, seed=101)

    # the same setup as the "still improving" case above, just given enough epochs to
    # actually reach 1.0 training accuracy (measured: epoch 6 of 20)
    result = train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=20)

    assert _training_accuracy(student, training_data) == 1.0

    diagnostic = result.diagnostic
    assert diagnostic.best_training_accuracy == 1.0
    assert diagnostic.converged is True
    assert diagnostic.plateaued is False
    assert diagnostic.still_improving is False


def test_train_linear_classifier_network_rejects_empty_training_data():

    bounds = square_bounds(10.0)
    student = LinearClassifierNetwork.randomized(1, 2, bounds)

    with pytest.raises(AssertionError):
        train_linear_classifier_network(student, [])


def test_train_linear_classifier_network_calls_a_schedule_with_increasing_step_indices():

    bounds = square_bounds(10.0)
    _reference, training_data = reachable_reference_and_training_data(
        1, 2, bounds, 5, rng=default_rng(0), data_rng=random.Random(0)
    )
    student = LinearClassifierNetwork.randomized(1, 2, bounds, seed=100)

    calls: list[int] = []

    def recording_schedule(step: int) -> float:
        calls.append(step)
        return 0.25

    epochs = 2
    train_linear_classifier_network(student, training_data, learning_rate=recording_schedule, epochs=epochs)

    assert calls == list(range(len(training_data) * epochs))


def test_training_and_its_data_leave_the_global_random_untouched():
    # every draw comes from a passed random.Random or one seeded from OS entropy (the RNG
    # generators workplan, D4, D6, D9), never the global stream a caller may have seeded
    random.seed(7)
    before = random.getstate()

    bounds = square_bounds(10.0)
    reference, training_data = reachable_reference_and_training_data(1, 2, bounds, 40)
    linear = LinearClassifierNetwork.randomized(1, 2, bounds)
    train_linear_classifier_network(linear, training_data, epochs=2, reference_classifier=reference)
    backprop = BackpropClassifierNetwork.randomized([3], 2, bounds)
    train_backprop_network_mini_batch(backprop, training_data, 8, epochs=2, reference_classifier=reference)

    assert random.getstate() == before
