from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.l2_regularized_backprop_classifier_network import L2RegularizedBackpropClassifierNetwork
from tests.helpers import (
    approx,
    assert_randomize_breaks_symmetry,
    assert_snapshot_restore_round_trip,
    wire_fixed_single_hidden_node,
)


def _fixed_network(l2_lambda: float = 0.1) -> L2RegularizedBackpropClassifierNetwork:
    # test_backprop_model.py's fixture
    network = L2RegularizedBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)], l2_lambda)
    wire_fixed_single_hidden_node(network)
    return network


def test_predict_probability_is_identical_to_the_plain_sgd_sibling():

    # L2 changes only apply_gradient: test_backprop_model.py's hand-derived a_o
    network = _fixed_network()

    assert network.predict_probability((2.0,)) == approx(0.5987376536170401)


def test_learn_matches_the_l2_regularized_update_rule_by_hand():

    # hand-derived with test_backprop_model.py's inputs and l2_lambda=0.1: the same deltas, the
    # weights decay by learning_rate * l2_lambda * w, and biases (never regularized) match plain
    # SGD's
    network = _fixed_network(l2_lambda=0.1)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == approx(0.49789010185038324)
    assert hidden_node.bias == approx(0.10144505092519163)
    assert output_node.input_node_weights[0] == approx(0.7992327797719059)
    assert output_node.bias == approx(-0.19035963698727032)


def test_l2_lambda_zero_matches_the_plain_sgd_sibling_bit_for_bit():

    l2_network = _fixed_network(l2_lambda=0.0)
    plain_network = BackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    wire_fixed_single_hidden_node(plain_network)

    l2_network.learn(0.1, (2.0,), 1.0)
    plain_network.learn(0.1, (2.0,), 1.0)

    l2_hidden = l2_network.hidden_layers[0].nodes[0]
    l2_output = l2_network.output_layer.nodes[0]
    plain_hidden = plain_network.hidden_layers[0].nodes[0]
    plain_output = plain_network.output_layer.nodes[0]
    assert l2_hidden.input_node_weights[0] == approx(plain_hidden.input_node_weights[0])
    assert l2_hidden.bias == approx(plain_hidden.bias)
    assert l2_output.input_node_weights[0] == approx(plain_output.input_node_weights[0])
    assert l2_output.bias == approx(plain_output.bias)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = L2RegularizedBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), l2_lambda=0.01)
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = L2RegularizedBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), l2_lambda=0.01)
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
