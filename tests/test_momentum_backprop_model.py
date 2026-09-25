import pytest
from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip, wire_fixed_single_hidden_node

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.momentum_backprop_classifier_network import MomentumBackpropClassifierNetwork


def _fixed_network(momentum: float = 0.9) -> MomentumBackpropClassifierNetwork:
    # test_backprop_model.py's fixture
    network = MomentumBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)], momentum)
    wire_fixed_single_hidden_node(network)
    return network


def test_first_learn_step_matches_the_plain_sgd_sibling_exactly():

    # with no prior step, momentum adds nothing: test_backprop_model.py's update-rule values
    network = _fixed_network(momentum=0.9)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.5028901018503833)
    assert hidden_node.bias == pytest.approx(0.10144505092519163)
    assert output_node.input_node_weights[0] == pytest.approx(0.8072327797719059)
    assert output_node.bias == pytest.approx(-0.19035963698727032)


def test_second_learn_step_shows_the_momentum_contribution_by_hand():

    # momentum first contributes on the second step. Hand-derived, momentum=0.9,
    # learning_rate=0.1, x=2.0, y=1.0:
    #   step 1 (plain SGD): a_h=0.7502601055951177, a_o=0.5987376536170401
    #   step 2: a_h=0.7516114513114047, a_o=0.6026132829266382
    network = _fixed_network(momentum=0.9)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)
    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.5083594576090832)
    assert hidden_node.bias == pytest.approx(0.10417972880454159)
    assert output_node.input_node_weights[0] == pytest.approx(0.8208947966338368)
    assert output_node.bias == pytest.approx(-0.17216707012974347)


def test_momentum_zero_matches_the_plain_sgd_sibling_across_many_steps():

    network = MomentumBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), momentum=0.0)
    hidden_node = network.hidden_layers[0].nodes[0]
    original_weights = list(hidden_node.input_node_weights)

    for _ in range(10):
        network.learn(0.1, (1.0, -2.0), 1.0)

    # momentum=0.0 still trains on the plain gradient; this checks it doesn't error or no-op
    assert hidden_node.input_node_weights != original_weights


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = MomentumBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), momentum=0.5)
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = MomentumBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), momentum=0.5)
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
