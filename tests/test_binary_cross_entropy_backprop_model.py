import pytest

from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip, wire_fixed_single_hidden_node
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.binary_cross_entropy_backprop_classifier_network import (
    BinaryCrossEntropyBackpropClassifierNetwork,
)


def _fixed_network() -> BinaryCrossEntropyBackpropClassifierNetwork:
    # test_backprop_model.py's fixture
    network = BinaryCrossEntropyBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    wire_fixed_single_hidden_node(network)
    return network


def test_predict_probability_is_identical_to_the_quadratic_loss_sibling():

    # cross-entropy changes only compute_output_delta, so the forward pass matches
    # test_backprop_model.py's hand-derived a_o
    network = _fixed_network()

    assert network.predict_probability((2.0,)) == pytest.approx(0.5987376536170401)


def test_learn_matches_the_binary_cross_entropy_update_rule_by_hand():

    # hand-derived, same inputs as test_backprop_model.py's update-rule test:
    #   delta_o = a_o - y = -0.4012623463829599 (no a_o*(1-a_o) factor: ~4x quadratic loss's
    #   -0.09640363012729687 here, hence cross-entropy's learning-rate sensitivity)
    #   delta_h = (delta_o * w_o) * a_h * (1 - a_h) = -0.06014758200698453
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
