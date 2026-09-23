import pytest

from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip, wire_fixed_single_hidden_node
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.binary_cross_entropy_backprop_classifier_network import (
    BinaryCrossEntropyBackpropClassifierNetwork,
)


def _fixed_network() -> BinaryCrossEntropyBackpropClassifierNetwork:
    # same dimension=1, one hidden node, one output node, and same starting weights as
    # test_backprop_model.py's own hand-computed fixture - deliberately, so the two loss
    # functions' forward pass (identical) and backward pass (different) can be compared side by
    # side from the exact same starting point (see the comments below)
    network = BinaryCrossEntropyBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    wire_fixed_single_hidden_node(network)
    return network


def test_predict_probability_is_identical_to_the_quadratic_loss_sibling():

    # cross-entropy only changes compute_output_delta - the forward pass (sigmoid activation)
    # is untouched, so predict_probability must match test_backprop_model.py's own hand-computed
    # forward-pass test exactly: z_h=1.1, a_h=sigmoid(1.1)=0.7502601055951177,
    # z_o=0.8*a_h-0.2=0.4002080844760941, a_o=sigmoid(z_o)=0.5987376536170401 (independently
    # computed, not re-derived from the implementation under test)
    network = _fixed_network()

    assert network.predict_probability((2.0,)) == pytest.approx(0.5987376536170401)


def test_learn_matches_the_binary_cross_entropy_update_rule_by_hand():

    # pins the backward-pass arithmetic against independently hand-derived expected values -
    # the direct binary-cross-entropy counterpart of
    # test_backprop_model.py::test_learn_matches_the_backprop_update_rule_by_hand, same starting
    # weights/state/learning_rate/category, so the two update rules' actual numeric divergence
    # is directly comparable.
    #   a_h = 0.7502601055951177, a_o = 0.5987376536170401 (same forward pass as the sibling)
    #   delta_o = a_o - y = -0.4012623463829599 (no a_o*(1-a_o) factor, unlike quadratic loss's
    #   delta_o=-0.09640363012729687 for this same state - cross-entropy's delta is ~4x larger
    #   here, the mechanism behind cross-entropy's learning-rate sensitivity)
    #   delta_h = (delta_o * w_o) * a_h * (1 - a_h) = -0.06014758200698453
    #   w -= learning_rate * delta * <that weight's input value>; b -= learning_rate * delta
    # state x=2.0, category y=1.0, learning_rate=0.1 - computed independently (not re-derived
    # from the implementation under test): new_w_h=0.5120295164013969,
    # new_b_h=0.10601475820069846, new_w_o=0.8301051130368624, new_b_o=-0.15987376536170403
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.5120295164013969)
    assert hidden_node.bias == pytest.approx(0.10601475820069846)
    assert output_node.input_node_weights[0] == pytest.approx(0.8301051130368624)
    assert output_node.bias == pytest.approx(-0.15987376536170403)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = BinaryCrossEntropyBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = BinaryCrossEntropyBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0))
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
