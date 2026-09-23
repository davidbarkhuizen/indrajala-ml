import pytest

from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip, wire_fixed_single_hidden_node
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.adam_backprop_classifier_network import AdamBackpropClassifierNetwork


def _fixed_network() -> AdamBackpropClassifierNetwork:
    # same dimension=1, one hidden node, one output node, and same starting weights as
    # test_backprop_model.py's own hand-computed fixture
    network = AdamBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    wire_fixed_single_hidden_node(network)
    return network


def test_first_learn_step_matches_a_near_pure_sign_step():

    # Adam's bias correction is exact at t=1 regardless of beta1/beta2 (m_hat == g, v_hat == g**2
    # when m/v start at zero), so the first update reduces to
    # w -= learning_rate * g / (sqrt(g**2) + epsilon), i.e. a near-pure per-parameter sign step
    # of magnitude ~learning_rate (deviating from an exact sign step only by epsilon/|g|, ~2e-7
    # relative here). Computed independently (not re-derived from the implementation under
    # test), using the same fixture as test_backprop_model.py's own hand-computed test (starting
    # weights w_h=0.5, b_h=0.1, w_o=0.8, b_o=-0.2; state x=2.0, category y=1.0,
    # learning_rate=0.1, Kingma & Ba's default beta1=0.9/beta2=0.999/epsilon=1e-8): every one of
    # the four raw gradients (g_wh=-0.028901018503832542, g_bh=-0.014450509251916271,
    # g_wo=-0.07232779771905842, g_bo=-0.09640363012729687) is negative, so every parameter moves
    # up by ~0.1: new_w_h=0.5999999653991552, new_b_h=0.1999999307983345,
    # new_w_o=0.8999999861740591, new_b_o=-0.10000001037305228.
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.5999999653991552)
    assert hidden_node.bias == pytest.approx(0.1999999307983345)
    assert output_node.input_node_weights[0] == pytest.approx(0.8999999861740591)
    assert output_node.bias == pytest.approx(-0.10000001037305228)


def test_second_learn_step_shows_real_moment_accumulation_by_hand():

    # two consecutive learn() calls on the same state/category, so the running mean/variance
    # estimates (m/v) actually accumulate across steps rather than each starting from zero -
    # computed independently (not re-derived from the implementation under test), continuing
    # from the first step's own m/v state: new_w_h=0.698656243383243,
    # new_b_h=0.2986561708026665, new_w_o=0.9994690924170146, new_b_o=-0.0009660949980676292.
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
