from pathlib import Path

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.softmax_multiclass_backprop_classifier_network import (
    SoftmaxMultiClassBackpropClassifierNetwork,
)
from tests.helpers import (
    approx,
    assert_randomize_breaks_symmetry,
    assert_save_and_load_round_trip,
    assert_snapshot_restore_round_trip,
)


def _fixed_network() -> SoftmaxMultiClassBackpropClassifierNetwork:
    # test_multiclass_backprop_model.py's fixture, so the two update rules are comparable
    network = SoftmaxMultiClassBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)], 2)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node_0, output_node_1 = network.output_layer.nodes
    hidden_node.update_input_weights([0.5])
    hidden_node.bias = 0.1
    output_node_0.update_input_weights([0.8])
    output_node_0.bias = -0.2
    output_node_1.update_input_weights([-0.3])
    output_node_1.bias = 0.4
    return network


def test_predict_probabilities_matches_a_hand_computed_softmax_forward_pass():

    # hand-derived: a_h = sigmoid(1.1) = 0.7502601055951177 (the hidden layer is still sigmoid)
    #   z_o0 = 0.8*a_h - 0.2 = 0.4002080844760942, z_o1 = -0.3*a_h + 0.4 = 0.17492196832146473
    #   softmax(z_o0, z_o1) = (0.5560845207454033, 0.4439154792545967)
    network = _fixed_network()

    probabilities = network.predict_probabilities((2.0,))

    assert probabilities[0] == approx(0.5560845207454033)
    assert probabilities[1] == approx(0.4439154792545967)
    assert sum(probabilities) == approx(1.0)


def test_classify_state_returns_the_argmax_class_index():

    network = _fixed_network()

    # output node 0's activation (0.556) > output node 1's (0.444) at this state
    assert network.classify_state((2.0,)) == 0


def test_learn_matches_the_softmax_cross_entropy_update_rule_by_hand():

    # hand-derived, x=2.0, category=1 (one-hot target (0, 1)), learning_rate=0.1:
    #   delta_o0 = a_o0 - 0.0 = 0.5560845207454033
    #   delta_o1 = a_o1 - 1.0 = -0.5560845207454033
    #   delta_h = (delta_o0*w_o0 + delta_o1*w_o1) * a_h * (1 - a_h) = 0.1146128386373376
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node_0, output_node_1 = network.output_layer.nodes

    network.learn(0.1, (2.0,), 1)

    assert hidden_node.input_node_weights[0] == approx(0.47707743227253246)
    assert hidden_node.bias == approx(0.08853871613626624)
    assert output_node_0.input_node_weights[0] == approx(0.7582791968745743)
    assert output_node_0.bias == approx(-0.25560845207454036)
    assert output_node_1.input_node_weights[0] == approx(-0.25827919687457435)
    assert output_node_1.bias == approx(0.45560845207454037)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = SoftmaxMultiClassBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), 3)
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = SoftmaxMultiClassBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), 4)
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 2))


def test_save_and_load_round_trip(tmp_path: Path):

    network = SoftmaxMultiClassBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), 4)
    for _ in range(5):
        network.learn(0.1, (1.0, -2.0), 2)

    loaded = assert_save_and_load_round_trip(
        network, SoftmaxMultiClassBackpropClassifierNetwork.load, tmp_path, "model.json", [(1.0, -2.0)]
    )

    assert isinstance(loaded, SoftmaxMultiClassBackpropClassifierNetwork)
    assert loaded.dimension == network.dimension
    assert loaded.class_count == network.class_count
    assert loaded.input_bounds == network.input_bounds

    # the loaded output layer is still softmax
    assert sum(loaded.predict_probabilities((1.0, -2.0))) == approx(1.0)
