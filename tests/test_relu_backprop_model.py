from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.relu_backprop_classifier_network import ReLUBackpropClassifierNetwork
from tests.helpers import (
    approx,
    assert_randomize_breaks_symmetry,
    assert_snapshot_restore_round_trip,
    wire_fixed_single_hidden_node,
)


def _fixed_network(x: float = 2.0) -> ReLUBackpropClassifierNetwork:
    # test_backprop_model.py's fixture
    network = ReLUBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    wire_fixed_single_hidden_node(network)
    return network


def test_predict_probability_matches_a_hand_computed_forward_pass():

    # hand-derived: z_h = 1.1, a_h = relu(1.1) = 1.1 (sigmoid would give 0.7502601055951177)
    #   z_o = 0.8*1.1 - 0.2 = 0.6800000000000002, a_o = sigmoid(z_o) = 0.6637386974043528
    network = _fixed_network()

    assert network.predict_probability((2.0,)) == approx(0.6637386974043528)


def test_learn_matches_the_relu_hidden_backprop_update_rule_by_hand():

    # hand-derived, same inputs as test_backprop_model.py's update-rule test:
    #   delta_o = (a_o - y) * a_o * (1 - a_o) = -0.0750500387266865
    #   delta_h = delta_o * w_o * 1.0 = -0.060040030981349204 (ReLU's derivative is 1 at a_h > 0,
    #   with no sigmoid a*(1-a) damping)
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == approx(0.5120080061962698)
    assert hidden_node.bias == approx(0.10600400309813493)
    assert output_node.input_node_weights[0] == approx(0.8082555042599355)
    assert output_node.bias == approx(-0.19249499612733137)


def test_learn_leaves_a_dead_units_incoming_weights_unchanged():

    # x=-10.0: z_h = -4.9, a_h = 0.0 (dead unit), so delta_h = 0 and the hidden weight and bias
    # don't move; the output weight's gradient is delta_o * a_h = 0, so it doesn't move either,
    # but the output bias does. Hand-derived: a_o = sigmoid(-0.2) = 0.45016600268752216,
    # delta_o = -0.13609302657524652
    network = _fixed_network(x=-10.0)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    assert network.predict_probability((-10.0,)) == approx(0.45016600268752216)

    network.learn(0.1, (-10.0,), 1.0)

    assert hidden_node.input_node_weights[0] == 0.5
    assert hidden_node.bias == 0.1
    assert output_node.input_node_weights[0] == 0.8
    assert output_node.bias == approx(-0.18639069734247535)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = ReLUBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = ReLUBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0))
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
