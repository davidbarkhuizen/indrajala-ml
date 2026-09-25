import pytest
from helpers import (
    assert_randomize_breaks_symmetry,
    assert_save_and_load_round_trip,
    assert_snapshot_restore_round_trip,
)

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork


def _fixed_network() -> MultiClassBackpropClassifierNetwork:
    # 1 input, 1 hidden node, 2 classes: small enough to derive by hand
    network = MultiClassBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)], 2)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node_0, output_node_1 = network.output_layer.nodes
    hidden_node.update_input_weights([0.5])
    hidden_node.bias = 0.1
    output_node_0.update_input_weights([0.8])
    output_node_0.bias = -0.2
    output_node_1.update_input_weights([-0.3])
    output_node_1.bias = 0.4
    return network


def test_predict_probabilities_matches_a_hand_computed_forward_pass():

    # hand-derived: z_h = 1.1, a_h = sigmoid(1.1) = 0.7502601055951177
    #   z_o0 = 0.8*a_h - 0.2, a_o0 = sigmoid(z_o0) = 0.5987376536170401
    #   z_o1 = -0.3*a_h + 0.4, a_o1 = sigmoid(z_o1) = 0.5436193278499907
    network = _fixed_network()

    probabilities = network.predict_probabilities((2.0,))

    assert probabilities[0] == pytest.approx(0.5987376536170401)
    assert probabilities[1] == pytest.approx(0.5436193278499907)


def test_classify_state_returns_the_argmax_class_index():

    network = _fixed_network()

    # output node 0's activation (0.599) > output node 1's (0.544) at this state
    assert network.classify_state((2.0,)) == 0


def test_learn_matches_the_one_vs_rest_update_rule_by_hand():

    # hand-derived, x=2.0, category=1 (one-hot target (0, 1)), learning_rate=0.1:
    #   delta_o0 = (a_o0 - 0.0) * a_o0 * (1 - a_o0)
    #   delta_o1 = (a_o1 - 1.0) * a_o1 * (1 - a_o1)
    #   delta_h = (delta_o0*w_o0 + delta_o1*w_o1) * a_h * (1 - a_h)
    #   w -= learning_rate * delta * <the weight's input>; b -= learning_rate * delta
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node_0, output_node_1 = network.output_layer.nodes

    network.learn(0.1, (2.0,), 1)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.49441465949423663)
    assert hidden_node.bias == pytest.approx(0.09720732974711832)
    assert output_node_0.input_node_weights[0] == pytest.approx(0.7892077150303392)
    assert output_node_0.bias == pytest.approx(-0.21438472456309046)
    assert output_node_1.input_node_weights[0] == pytest.approx(-0.29150504211018)
    assert output_node_1.bias == pytest.approx(0.4113226837285739)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = MultiClassBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), 3)
    assert_randomize_breaks_symmetry(network)


def test_randomize_scales_weight_range_with_fan_in():

    # limit = 1/sqrt(fan_in): a wider hidden layer gives the output layer a narrower range
    narrow = MultiClassBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), 3)
    wide = MultiClassBackpropClassifierNetwork.randomized([400], 2, square_bounds(10.0), 3)

    narrow_output_range = max(abs(w) for node in narrow.output_layer.nodes for w in node.input_node_weights)
    wide_output_range = max(abs(w) for node in wide.output_layer.nodes for w in node.input_node_weights)

    assert wide_output_range < narrow_output_range


def test_snapshot_and_restore_round_trip():

    network = MultiClassBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), 4)
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 2))


def test_save_and_load_round_trip(tmp_path):

    network = MultiClassBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), 4)
    for _ in range(5):
        network.learn(0.1, (1.0, -2.0), 2)

    loaded = assert_save_and_load_round_trip(
        network,
        MultiClassBackpropClassifierNetwork.load,
        tmp_path,
        "model.json",
        [(1.0, -2.0), (-3.0, 4.0), (0.0, 0.0)],
    )

    assert loaded.dimension == network.dimension
    assert loaded.class_count == network.class_count
    assert loaded.input_bounds == network.input_bounds
