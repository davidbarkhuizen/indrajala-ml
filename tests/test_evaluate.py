import random

import pytest

from indrajala_ml.evaluate import (
    class_balanced_disagreement_rate,
    compare_on_random_point,
    sample_class_balanced_states,
    smoothed_series,
)
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.train import reachable_reference_and_training_data

from helpers import classifier_with_tiny_bounded_region, unreachable_class_classifier


def test_sample_class_balanced_states_succeeds_within_a_tight_budget_for_a_tiny_region():

    # would reliably exhaust a budget this small under naive full-input_bounds rejection
    # sampling (median cardinality=4 bounded region covers ~1.5% of the box; this one is
    # deliberately far smaller, at 0.01% - verified directly: naive sampling found only 1/20
    # positive points in 5000 attempts, where sampling from a tight box around the region
    # instead found 20/20 in 28) - succeeds here because positive-class points are drawn from
    # a tight box around the region itself, not the whole box
    bounds = square_bounds(10.0)
    classifier = classifier_with_tiny_bounded_region(bounds)

    positive_states, negative_states = sample_class_balanced_states(classifier, count=20, max_attempts=1000)

    assert len(positive_states) == len(negative_states) == 20
    assert all(classifier.classify_state(state) == 1.0 for state in positive_states)
    assert all(classifier.classify_state(state) == 0.0 for state in negative_states)


def test_sample_class_balanced_states_falls_back_to_input_bounds_when_unbounded():

    # cardinality=1 is never bounded, so positive_region_bounding_box returns None - this
    # should still work exactly as before (uniform sampling over the whole input_bounds)
    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork.randomized(1, 2, bounds)

    positive_states, negative_states = sample_class_balanced_states(classifier, count=10, max_attempts=20_000)

    assert len(positive_states) == len(negative_states) == 10
    assert all(classifier.classify_state(state) == 1.0 for state in positive_states)
    assert all(classifier.classify_state(state) == 0.0 for state in negative_states)


def test_class_balanced_disagreement_rate_is_zero_for_identical_classifier():

    classifier, _ = reachable_reference_and_training_data(2, 2, square_bounds(5.0), 10)

    assert class_balanced_disagreement_rate(classifier, classifier) == 0.0


def test_class_balanced_disagreement_rate_detects_error_the_old_metric_missed():

    random.seed(1)

    cardinality, dimension, l = 4, 2, 10.0
    bounds = square_bounds(l, dimension)

    reference = LinearClassifierNetwork.randomized(cardinality, dimension, bounds)
    student = LinearClassifierNetwork.randomized(cardinality, dimension, bounds)

    assert class_balanced_disagreement_rate(reference, student, per_class_sample_count=200) > 0.3


def test_class_balanced_disagreement_rate_raises_for_an_unreachable_reference_class():

    unreachable = unreachable_class_classifier(square_bounds(10.0))

    with pytest.raises(RuntimeError):
        class_balanced_disagreement_rate(unreachable, unreachable, per_class_sample_count=10, max_attempts=200)


def test_class_balanced_disagreement_rate_rejects_a_zero_sample_count():

    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork.randomized(1, 2, bounds)

    with pytest.raises(AssertionError):
        class_balanced_disagreement_rate(classifier, classifier, per_class_sample_count=0)


def test_compare_on_random_point_classifies_with_both_networks():

    classifier, _ = reachable_reference_and_training_data(2, 2, square_bounds(5.0), 10)

    state, reference_category, student_category = compare_on_random_point(classifier, classifier)

    assert len(state) == 2
    assert all(bound[0] <= value <= bound[1] for value, bound in zip(state, classifier.input_bounds))
    assert reference_category == student_category == classifier.classify_state(state)


def test_smoothed_series_with_window_one_is_identity():

    values = [0.0, 1.0, 0.0, 1.0]

    assert smoothed_series(values, window=1) == values


def test_smoothed_series_is_a_trailing_moving_average():

    values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    smoothed = smoothed_series(values, window=3)

    assert len(smoothed) == len(values)
    assert smoothed[0] == 0.0
    assert smoothed[1] == 0.5
    assert smoothed[2] == pytest.approx(1 / 3)
    assert smoothed[3] == pytest.approx(2 / 3)
