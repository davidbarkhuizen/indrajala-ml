import pytest

from indrajala_ml.geometry import (
    is_positive_region_bounded,
    positive_region_bounding_box,
    reference_positive_region_polygon,
    square_bounds,
)
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork

from helpers import classifier_with_bounded_square_region


def test_square_bounds():

    assert square_bounds(10.0) == [(-10.0, 10.0), (-10.0, 10.0)]
    assert square_bounds(10.0, dimension=3) == [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]


def test_is_positive_region_bounded_true_for_a_known_bounded_square():

    bounds = square_bounds(10.0)
    classifier = classifier_with_bounded_square_region(bounds)

    assert is_positive_region_bounded(classifier) is True


def test_is_positive_region_bounded_false_for_cardinality_one():

    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork.randomized(1, 2, bounds)

    # a single half-plane can never be a bounded region
    assert is_positive_region_bounded(classifier) is False


def test_is_positive_region_bounded_false_for_a_genuinely_empty_region():

    # unlike the unbounded half-plane above, x > 5 and x < -5 intersect in nothing
    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork(2, 2, bounds)
    greater_than_five, less_than_negative_five = classifier.hidden_layer.nodes
    greater_than_five.update_input_weights([1.0, 0.0])
    greater_than_five.threshold = -5.0
    less_than_negative_five.update_input_weights([-1.0, 0.0])
    less_than_negative_five.threshold = -5.0

    assert reference_positive_region_polygon(classifier) == []
    assert is_positive_region_bounded(classifier) is False


def test_reference_positive_region_polygon_rejects_a_non_and_classifier():

    # the intersection is the positive region only under AND; otherwise the region is a union
    # of intersections, so the function must refuse, not return a wrong polygon
    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork(2, 2, bounds, required_active=1)

    with pytest.raises(AssertionError):
        reference_positive_region_polygon(classifier)

    with pytest.raises(AssertionError):
        is_positive_region_bounded(classifier)


def test_reference_positive_region_polygon_rejects_a_non_2d_classifier():

    # it reads only the first two weights, so dimension > 2 would silently give a wrong answer
    bounds = [(-10.0, 10.0)] * 3
    classifier = LinearClassifierNetwork(4, 3, bounds)
    for node, (weights, threshold) in zip(
        classifier.hidden_layer.nodes,
        [([1.0, 0.0, 0.0], 1.0), ([-1.0, 0.0, 0.0], 1.0), ([0.0, 1.0, 0.0], 1.0), ([0.0, -1.0, 0.0], 1.0)],
    ):
        node.update_input_weights(weights)
        node.threshold = threshold

    with pytest.raises(AssertionError):
        reference_positive_region_polygon(classifier)

    with pytest.raises(AssertionError):
        is_positive_region_bounded(classifier)


def test_is_positive_region_bounded_true_for_a_large_scale_bounded_region():

    l = 1.0e7
    bounds = square_bounds(l)
    classifier = LinearClassifierNetwork(4, 2, bounds)
    half_extent = 2.0e6
    for node, (weights, threshold) in zip(
        classifier.hidden_layer.nodes,
        [
            ([1.0, 0.0], half_extent),
            ([-1.0, 0.0], half_extent),
            ([0.0, 1.0], half_extent),
            ([0.0, -1.0], half_extent),
        ],
    ):
        node.update_input_weights(weights)
        node.threshold = threshold

    assert is_positive_region_bounded(classifier) is True


def test_positive_region_bounding_box_is_tight_around_a_known_bounded_square():

    # positive region is exactly [-1, 1] x [-1, 1] - the tight box (with a small margin)
    # should stay far smaller than the full input_bounds, not just echo it back
    bounds = square_bounds(10.0)
    classifier = classifier_with_bounded_square_region(bounds)

    box = positive_region_bounding_box(classifier)

    assert box is not None
    (x_min, x_max), (y_min, y_max) = box
    assert -1.5 < x_min < -1.0 and 1.0 < x_max < 1.5
    assert -1.5 < y_min < -1.0 and 1.0 < y_max < 1.5


def test_positive_region_bounding_box_none_when_unbounded():

    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork.randomized(1, 2, bounds)

    assert positive_region_bounding_box(classifier) is None


def test_positive_region_bounding_box_none_for_a_non_and_classifier():

    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork(2, 2, bounds, required_active=1)

    assert positive_region_bounding_box(classifier) is None


def test_positive_region_bounding_box_none_for_a_non_2d_classifier():

    bounds = [(-10.0, 10.0)] * 3
    classifier = LinearClassifierNetwork.randomized(1, 3, bounds)

    assert positive_region_bounding_box(classifier) is None


def test_positive_region_bounding_box_none_for_a_duck_typed_target_without_geometry_attributes():

    # targets need only input_bounds and classify_state (e.g. targets.py's XORTarget); without
    # the linear classifier's geometry, fall back to None, not AttributeError
    class BoundsOnly:
        def __init__(self, bounds):
            self.input_bounds = bounds

        def classify_state(self, state):
            return 1.0

    assert positive_region_bounding_box(BoundsOnly(square_bounds(10.0))) is None
