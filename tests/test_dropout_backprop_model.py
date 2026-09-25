from unittest.mock import patch

import pytest

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.dropout_backprop_classifier_network import DropoutBackpropClassifierNetwork
from indrajala_ml.model.dropout_layer import TrainingModeNode
from tests.helpers import (
    approx,
    assert_randomize_breaks_symmetry,
    assert_snapshot_restore_round_trip,
    wire_fixed_single_hidden_node,
)


def _fixed_network(drop_probability: float = 0.5) -> DropoutBackpropClassifierNetwork:
    # test_backprop_model.py's fixture
    network = DropoutBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)], drop_probability)
    wire_fixed_single_hidden_node(network)
    return network


def test_predict_probability_at_eval_mode_matches_the_plain_sigmoid_baseline_exactly():

    # test_backprop_model.py's hand-derived a_o: dropout is a no-op at inference
    network = _fixed_network()

    assert network.predict_probability((2.0,)) == approx(0.5987376536170401)


def test_predict_probability_is_deterministic_run_to_run_no_stochasticity_at_inference():

    network = _fixed_network()

    predictions = [network.predict_probability((2.0,)) for _ in range(20)]

    assert len(set(predictions)) == 1


def test_learn_when_the_hidden_unit_is_kept_matches_the_inverted_dropout_update_rule_by_hand():

    # hand-derived, forced kept (0.9 >= 0.5):
    #   a_h = sigmoid(1.1) / 0.5 = 1.5005202111902354 (inverted-dropout rescale)
    #   z_o = 0.8*a_h - 0.2 = 1.0004161689521884, a_o = sigmoid(z_o) = 0.7311403945436992
    #   delta_o = (a_o - 1) * a_o * (1 - a_o) = -0.052850839811138146
    #   delta_h = delta_o * w_o * base * (1 - base) / 0.5 = -0.015844248783037213, where base =
    #   sigmoid(1.1) is the unscaled activation (see compute_hidden_delta in dropout_layer.py)
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    with patch("random.random", return_value=0.9):
        network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == approx(0.5031688497566075)
    assert hidden_node.bias == approx(0.10158442487830373)
    assert output_node.input_node_weights[0] == approx(0.807930375331499)
    assert output_node.bias == approx(-0.19471491601888619)


def test_learn_when_the_hidden_unit_is_dropped_leaves_its_incoming_weights_unchanged():

    # forced dropped (0.1 < 0.5): a_h = 0.0, so only the output bias moves, to the same value as
    # test_relu_backprop_model.py's dead unit
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    with patch("random.random", return_value=0.1):
        network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == 0.5
    assert hidden_node.bias == 0.1
    assert output_node.input_node_weights[0] == 0.8
    assert output_node.bias == approx(-0.18639069734247535)


def test_predict_probability_between_learn_calls_is_unaffected_by_training_mode():

    # learn() must not leave training mode on for the next prediction
    network = _fixed_network()

    with patch("random.random", return_value=0.1):  # would drop, if this leaked into eval mode
        network.learn(0.1, (2.0,), 1.0)

    prediction = network.predict_probability((2.0,))

    assert 0.0 < prediction < 1.0


def test_learn_batch_draws_an_independent_mask_per_example_not_one_shared_per_batch():

    # training mode spans the whole batch, but each example still draws its own mask
    network = DropoutBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), drop_probability=0.5)
    batch = [((1.0, -1.0), 1.0), ((-1.0, 1.0), 0.0), ((0.5, 0.5), 1.0)]

    with patch("random.random", return_value=0.9) as mock_random:
        network.learn_batch(0.1, batch)

    assert mock_random.call_count == len(batch) * 4  # 4 hidden nodes, one draw each per example


def test_learn_batch_leaves_training_mode_off_afterward():

    network = DropoutBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), drop_probability=0.5)
    batch = [((1.0, -1.0), 1.0), ((-1.0, 1.0), 0.0)]

    network.learn_batch(0.1, batch)

    nodes = network.hidden_layers[0].nodes
    assert all(isinstance(node, TrainingModeNode) and not node.training for node in nodes)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = DropoutBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), drop_probability=0.5)
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = DropoutBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), drop_probability=0.5)
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))


def test_drop_probability_is_a_required_constructor_argument():

    with pytest.raises(TypeError):
        DropoutBackpropClassifierNetwork([4], 2, square_bounds(10.0))  # type: ignore[call-arg]
