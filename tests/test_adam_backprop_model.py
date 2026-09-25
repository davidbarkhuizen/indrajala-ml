import pytest

from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip, wire_fixed_single_hidden_node
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.adam_backprop_classifier_network import AdamBackpropClassifierNetwork


def _fixed_network() -> AdamBackpropClassifierNetwork:
    # test_backprop_model.py's fixture
    network = AdamBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    wire_fixed_single_hidden_node(network)
    return network


def test_first_learn_step_matches_a_near_pure_sign_step():

    # at t=1 bias correction gives m_hat = g and v_hat = g**2, so the step is
    # learning_rate * g / (|g| + epsilon): a sign step of ~learning_rate. Hand-derived with
    # test_backprop_model.py's inputs and beta1=0.9, beta2=0.999, epsilon=1e-8: all four
    # gradients (g_wh=-0.028901018503832542, g_bh=-0.014450509251916271,
    # g_wo=-0.07232779771905842, g_bo=-0.09640363012729687) are negative, so each parameter
    # rises by ~0.1
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.5999999653991552)
    assert hidden_node.bias == pytest.approx(0.1999999307983345)
    assert output_node.input_node_weights[0] == pytest.approx(0.8999999861740591)
    assert output_node.bias == pytest.approx(-0.10000001037305228)


def test_second_learn_step_shows_real_moment_accumulation_by_hand():

    # hand-derived: a second step on the same example, carrying the first step's m and v
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)
    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.698656243383243)
    assert hidden_node.bias == pytest.approx(0.2986561708026665)
    assert output_node.input_node_weights[0] == pytest.approx(0.9994690924170146)
    assert output_node.bias == pytest.approx(-0.0009660949980676292)


def test_trains_over_many_steps_without_erroring_or_stalling():

    network = AdamBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    hidden_node = network.hidden_layers[0].nodes[0]
    original_weights = list(hidden_node.input_node_weights)

    for _ in range(10):
        network.learn(0.1, (1.0, -2.0), 1.0)

    assert hidden_node.input_node_weights != original_weights


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = AdamBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = AdamBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0))
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))
