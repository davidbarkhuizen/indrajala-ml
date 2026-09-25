import random

import pytest

from indrajala_ml.geometry import is_positive_region_bounded, square_bounds
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.targets import XORTarget
from indrajala_ml.train import (
    random_alternating_training_data,
    reachable_reference_and_training_data,
    train_linear_classifier_network,
)

from helpers import classifier_with_tiny_bounded_region, unreachable_class_classifier


def _training_accuracy(student, training_data):
    return sum(1 for state, category in training_data if student.classify_state(state) == category) / len(
        training_data
    )


def test_generation_of_random_test_data_from_reference_classifier():

    classifier_cardinality = 1
    dimension: int = 2
    l: float = 7.0
    training_set_size: int = 50

    input_bounds = square_bounds(l, dimension)

    _, training_data = reachable_reference_and_training_data(
        classifier_cardinality, dimension, input_bounds, training_set_size
    )

    assert len(training_data) == training_set_size


def test_generation_of_random_test_data_for_non_2d_classifier():

    dimension: int = 3
    training_set_size: int = 50
    input_bounds = [(-7.0, 7.0)] * dimension

    _, training_data = reachable_reference_and_training_data(1, dimension, input_bounds, training_set_size)

    assert len(training_data) == training_set_size
    assert all(len(state) == dimension for state, _ in training_data)


def test_reachable_reference_and_training_data_returns_class_balanced_data():

    cardinality = 4
    dimension = 2
    bounds = square_bounds(10.0)
    training_set_size = 200

    reference, training_data = reachable_reference_and_training_data(cardinality, dimension, bounds, training_set_size)

    assert reference.cardinality == cardinality
    assert len(training_data) == training_set_size
    assert all(len(state) == dimension for state, _ in training_data)


def test_reachable_reference_and_training_data_respects_is_valid():

    cardinality = 4
    dimension = 2
    bounds = square_bounds(10.0)

    reference, _ = reachable_reference_and_training_data(
        cardinality, dimension, bounds, 200, regeneration_attempts=200, is_valid=is_positive_region_bounded
    )

    assert is_positive_region_bounded(reference) is True


def test_random_alternating_training_data_raises_for_an_unreachable_class():

    unreachable = unreachable_class_classifier(square_bounds(10.0))

    with pytest.raises(RuntimeError):
        random_alternating_training_data(200, unreachable, max_attempts=200)


def test_random_alternating_training_data_succeeds_within_a_tight_budget_for_a_tiny_region():

    # via evaluate.sample_class_balanced_states: naive rejection sampling would exhaust this
    # budget on a region 0.01% of the box
    bounds = square_bounds(10.0)
    classifier = classifier_with_tiny_bounded_region(bounds)

    training_data = random_alternating_training_data(40, classifier, max_attempts=1000)

    assert len(training_data) == 40
    assert sum(1 for _, category in training_data if category == 1.0) == 20
    assert sum(1 for _, category in training_data if category == 0.0) == 20


def test_reachable_reference_and_training_data_raises_when_no_reference_is_ever_valid():

    with pytest.raises(RuntimeError):
        reachable_reference_and_training_data(
            1, 2, square_bounds(10.0), 50, regeneration_attempts=5, is_valid=lambda classifier: False
        )


def test_train_linear_classifier_network_keeps_the_best_epoch_not_the_last():

    # measured per-epoch accuracy without pocket tracking: 0.623, 0.710, 0.807, 0.845, 0.830,
    # 0.816, 0.801, 0.843, 0.827, 0.829, so the best epoch (3) isn't the last
    random.seed(0)

    # XOR isn't representable by an AND/OR/k-of-n gate over cardinality=3 half-planes
    bounds = square_bounds(10.0)
    training_data = random_alternating_training_data(1000, XORTarget(bounds))

    student = LinearClassifierNetwork.randomized(3, 2, bounds, required_active=2)
    result = train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=10)

    assert _training_accuracy(student, training_data) == pytest.approx(0.845)

    # best epoch index 3, last 9: a plateau
    diagnostic = result.diagnostic
    assert diagnostic.epoch_training_accuracies == [
        pytest.approx(a) for a in [0.623, 0.710, 0.807, 0.845, 0.830, 0.816, 0.801, 0.843, 0.827, 0.829]
    ]
    assert diagnostic.best_epoch_index == 3
    assert diagnostic.best_training_accuracy == pytest.approx(0.845)
    assert diagnostic.plateaued is True
    assert diagnostic.converged is False
    assert diagnostic.still_improving is False


def test_train_linear_classifier_network_pocket_tracking_is_a_no_op_when_it_converges():

    # when training does converge, the best epoch and the last epoch coincide, so pocket
    # tracking shouldn't change the well-established convergence behavior at all
    random.seed(6)

    bounds = square_bounds(10.0)
    reference, training_data = reachable_reference_and_training_data(1, 2, bounds, 400)
    student = LinearClassifierNetwork.randomized(1, 2, bounds)

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

    random.seed(0)

    bounds = square_bounds(10.0)
    reference, training_data = reachable_reference_and_training_data(1, 2, bounds, 400)
    student = LinearClassifierNetwork.randomized(1, 2, bounds)

    # the same setup as the "still improving" case above, just given enough epochs to
    # actually reach 1.0 training accuracy (measured: epoch 16 of 20)
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

    random.seed(0)

    bounds = square_bounds(10.0)
    reference, training_data = reachable_reference_and_training_data(1, 2, bounds, 5)
    student = LinearClassifierNetwork.randomized(1, 2, bounds)

    calls: list[int] = []

    def recording_schedule(step: int) -> float:
        calls.append(step)
        return 0.25

    epochs = 2
    train_linear_classifier_network(student, training_data, learning_rate=recording_schedule, epochs=epochs)

    assert calls == list(range(len(training_data) * epochs))
