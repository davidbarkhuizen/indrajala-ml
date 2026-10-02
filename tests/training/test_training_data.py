import pytest

from indrajala_ml.geometry import is_positive_region_bounded, square_bounds
from indrajala_ml.training.training_data import random_alternating_training_data, reachable_reference_and_training_data
from tests.helpers import classifier_with_tiny_bounded_region, unreachable_class_classifier


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
