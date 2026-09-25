import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


from indrajala_ml.geometry import square_bounds
from indrajala_ml.graphics.chart import (
    disagreement_axis_bounds,
    new_axes,
    new_figure,
    plot_linear_classifier_network,
    plot_training_data,
    reference_region_bounds,
)
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from tests.helpers import classifier_with_bounded_square_region


def test_reference_region_bounds_expands_to_include_a_bounded_region():

    bounds = square_bounds(10.0)
    classifier = classifier_with_bounded_square_region(bounds)

    small_fallback = [(-0.5, 0.5), (-0.5, 0.5)]
    (x_min, x_max), (y_min, y_max) = reference_region_bounds(classifier, small_fallback)

    assert x_min < -1.0 and x_max > 1.0
    assert y_min < -1.0 and y_max > 1.0


def test_reference_region_bounds_leaves_fallback_unchanged_when_unbounded():

    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork.randomized(1, 2, bounds)

    assert reference_region_bounds(classifier, bounds) == bounds


def test_disagreement_axis_bounds():

    assert disagreement_axis_bounds(500.0) == [(0.0, 500.0), (0.0, 1.0)]
    assert disagreement_axis_bounds(500.0, log=True) == [(0.0, 500.0), (1.0e-3, 1.0)]


def test_plot_training_data_assigns_marker_and_color_by_sorted_category_not_set_order():

    # category-to-marker/color assignment must go by sorted category order, not iteration
    # order over a set(). Pins down that class 0.0 always gets the first marker/color
    # ("." / blue) and 1.0 the second ("x" / yellow).
    bounds = square_bounds(10.0)
    training_data = [((1.0, 1.0), 1.0), ((-1.0, -1.0), 0.0), ((2.0, 2.0), 1.0), ((-2.0, -2.0), 0.0)]

    axes = new_axes(new_figure("test"), bounds)
    plot_training_data(axes, training_data)

    assert len(axes.lines) == 2
    zero_line, one_line = axes.lines
    assert zero_line.get_marker() == "." and zero_line.get_color() == "blue"
    assert np.asarray(zero_line.get_xdata()).tolist() == [-1.0, -2.0]
    assert one_line.get_marker() == "x" and one_line.get_color() == "yellow"
    assert np.asarray(one_line.get_xdata()).tolist() == [1.0, 2.0]


def test_plot_linear_classifier_network_draws_a_vertical_line_for_a_zero_y_weight():

    # a*x + c = 0 (no y term) can't be solved for y as a function of x; should draw a
    # vertical line at x = -c/a instead
    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork(1, 2, bounds)
    node = classifier.hidden_layer.nodes[0]
    node.update_input_weights([1.0, 0.0])
    node.threshold = -5.0

    axes = new_axes(new_figure("test"), bounds)
    plot_linear_classifier_network(axes, classifier)

    assert len(axes.lines) == 1
    line = axes.lines[0]
    assert np.asarray(line.get_xdata()).tolist() == [5.0, 5.0]
    assert np.asarray(line.get_ydata()).tolist() == list(bounds[1])


def test_plot_linear_classifier_network_skips_a_node_with_no_weights_at_all():

    # a*x + b*y + c = 0 with a == b == 0 doesn't depend on position at all (the node is
    # either always active or always inactive everywhere) - there's no line to draw
    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork(1, 2, bounds)
    node = classifier.hidden_layer.nodes[0]
    node.update_input_weights([0.0, 0.0])
    node.threshold = 1.0

    axes = new_axes(new_figure("test"), bounds)
    plot_linear_classifier_network(axes, classifier)

    assert len(axes.lines) == 0


def test_plot_linear_classifier_network_rejects_a_non_2d_classifier():

    bounds = [(-10.0, 10.0)] * 3
    classifier = LinearClassifierNetwork.randomized(1, 3, bounds)

    axes = new_axes(new_figure("test"), square_bounds(10.0))

    with pytest.raises(AssertionError):
        plot_linear_classifier_network(axes, classifier)
