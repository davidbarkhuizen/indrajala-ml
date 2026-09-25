from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from tests.helpers import (
    approx,
    assert_randomize_breaks_symmetry,
    assert_snapshot_restore_round_trip,
    wire_fixed_single_hidden_node,
)


def _fixed_network() -> BackpropClassifierNetwork:
    # 1 input, 1 hidden node, 1 output: small enough to derive by hand. The *_backprop_model.py
    # siblings share these weights, so their results are directly comparable
    network = BackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    wire_fixed_single_hidden_node(network)
    return network


def test_predict_probability_matches_a_hand_computed_forward_pass():

    # hand-derived, w_h=0.5, b_h=0.1, w_o=0.8, b_o=-0.2, x=2.0:
    #   z_h = 1.1, a_h = sigmoid(1.1) = 0.7502601055951177
    #   z_o = 0.8*a_h - 0.2 = 0.4002080844760941, a_o = sigmoid(z_o) = 0.5987376536170401
    network = _fixed_network()

    assert network.predict_probability((2.0,)) == approx(0.5987376536170401)


def test_classify_state_thresholds_strictly_above_half():

    network = BackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]
    hidden_node.update_input_weights([0.0])
    hidden_node.bias = 0.0
    output_node.update_input_weights([0.0])
    output_node.bias = 0.0

    # z_o = 0.0 -> a_o = sigmoid(0) = 0.5 exactly - must not classify as active
    assert network.predict_probability((0.0,)) == approx(0.5)
    assert network.classify_state((0.0,)) == 0.0


def test_learn_matches_the_backprop_update_rule_by_hand():

    # hand-derived, x=2.0, y=1.0, learning_rate=0.1:
    #   delta_o = (a_o - y) * a_o * (1 - a_o) = -0.09640363012729687
    #   delta_h = (delta_o * w_o) * a_h * (1 - a_h) = -0.014450509251916271
    #   w -= learning_rate * delta * <the weight's input>; b -= learning_rate * delta
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == approx(0.5028901018503833)
    assert hidden_node.bias == approx(0.10144505092519163)
    assert output_node.input_node_weights[0] == approx(0.8072327797719059)
    assert output_node.bias == approx(-0.19035963698727032)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = BackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0))
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
