from unittest.mock import patch

import pytest

from helpers import assert_randomize_breaks_symmetry, assert_snapshot_restore_round_trip, wire_fixed_single_hidden_node
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.dropout_backprop_classifier_network import DropoutBackpropClassifierNetwork


def _fixed_network(drop_probability: float = 0.5) -> DropoutBackpropClassifierNetwork:
    # same dimension=1, one hidden node, one output node, and same starting weights as
    # test_backprop_model.py's own hand-computed fixture - deliberately, so predict_probability
    # at eval mode (dropout inactive) can be checked against the exact same baseline number
    network = DropoutBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)], drop_probability)
    wire_fixed_single_hidden_node(network)
    return network


def test_predict_probability_at_eval_mode_matches_the_plain_sigmoid_baseline_exactly():

    # z_h=1.1, a_h=sigmoid(1.1)=0.7502601055951177, z_o=0.8*a_h-0.2=0.4002080844760941,
    # a_o=sigmoid(z_o)=0.5987376536170401 - the exact same numbers
    # test_backprop_model.py's own baseline fixture produces, confirming dropout is a genuine
    # no-op at inference, not just "usually close"
    network = _fixed_network()

    assert network.predict_probability((2.0,)) == pytest.approx(0.5987376536170401)


def test_predict_probability_is_deterministic_run_to_run_no_stochasticity_at_inference():

    # the one genuinely new property no prior sibling's test suite needed to check: no prior
    # sibling has any training-only behavior, so nothing else here could ever produce a
    # different prediction for the same weights on repeated calls
    network = _fixed_network()

    predictions = [network.predict_probability((2.0,)) for _ in range(20)]

    assert len(set(predictions)) == 1


def test_learn_when_the_hidden_unit_is_kept_matches_the_inverted_dropout_update_rule_by_hand():

    # forced kept (random.random()=0.9 >= drop_probability=0.5): a_h = sigmoid(1.1)/0.5 =
    # 1.5005202111902354 (inverted-dropout rescale), z_o = 0.8*a_h - 0.2 = 1.0004161689521884,
    # a_o = sigmoid(z_o) = 0.7311403945436992, delta_o = (a_o-1)*a_o*(1-a_o) = -0.052850839811138146
    # delta_h = (delta_o*w_o) * base*(1-base) / keep_probability = -0.015844248783037213, where
    # base=sigmoid(1.1)=0.7502601055951177 is the *unscaled* activation (see dropout_layer.py's
    # own compute_hidden_delta docstring for why base, not a_h, is the correct derivative term)
    # computed independently (not re-derived from the implementation under test):
    # new_w_h=0.5031688497566075, new_b_h=0.10158442487830373, new_w_o=0.807930375331499,
    # new_b_o=-0.19471491601888619
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    with patch("random.random", return_value=0.9):
        network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == pytest.approx(0.5031688497566075)
    assert hidden_node.bias == pytest.approx(0.10158442487830373)
    assert output_node.input_node_weights[0] == pytest.approx(0.807930375331499)
    assert output_node.bias == pytest.approx(-0.19471491601888619)


def test_learn_when_the_hidden_unit_is_dropped_leaves_its_incoming_weights_unchanged():

    # forced dropped (random.random()=0.1 < drop_probability=0.5): a_h=0.0, delta_h=0.0 exactly
    # (the dropout-dead-unit case - matches ReLUNode's own dead-unit precedent, and lands on the
    # exact same downstream numbers test_relu_backprop_model.py's own
    # test_learn_leaves_a_dead_units_incoming_weights_unchanged does, since a_h=0.0 produces an
    # identical z_o regardless of *why* it's zero: z_o=-0.2, a_o=sigmoid(-0.2)=0.45016600268752216,
    # delta_o=-0.13609302657524652, new_b_o=-0.18639069734247535 - computed independently (not
    # re-derived from the implementation under test)
    network = _fixed_network()
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]

    with patch("random.random", return_value=0.1):
        network.learn(0.1, (2.0,), 1.0)

    assert hidden_node.input_node_weights[0] == 0.5
    assert hidden_node.bias == 0.1
    assert output_node.input_node_weights[0] == 0.8
    assert output_node.bias == pytest.approx(-0.18639069734247535)


def test_predict_probability_between_learn_calls_is_unaffected_by_training_mode():

    # call-scoped, not lifecycle-scoped: a predict_probability() call sandwiched between two
    # learn() calls must see eval-mode behavior, not accidentally inherit training mode left on
    # by the learn() call before it
    network = _fixed_network()

    with patch("random.random", return_value=0.1):  # would drop, if this leaked into eval mode
        network.learn(0.1, (2.0,), 1.0)

    prediction = network.predict_probability((2.0,))

    # eval-mode is unconditional (self.training defaults to False on every fresh forward pass
    # unless set_training_mode(True) is currently bracketing it) - a real probability in (0, 1),
    # not the 0.0 a leaked training-mode-dropped hidden unit would force
    assert 0.0 < prediction < 1.0


def test_learn_batch_draws_an_independent_mask_per_example_not_one_shared_per_batch():

    # _learn_batch brackets set_training_mode(True) around the whole batch loop (unlike learn(),
    # which only brackets the single forward() call) - confirms that doesn't collapse into "one
    # mask decided once for the whole batch": every example's forward pass draws its own mask,
    # one random.random() call per hidden node per example
    network = DropoutBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), drop_probability=0.5)
    batch = [((1.0, -1.0), 1.0), ((-1.0, 1.0), 0.0), ((0.5, 0.5), 1.0)]

    with patch("random.random", return_value=0.9) as mock_random:
        network.learn_batch(0.1, batch)

    assert mock_random.call_count == len(batch) * 4  # 4 hidden nodes, one draw each per example


def test_learn_batch_leaves_training_mode_off_afterward():

    network = DropoutBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), drop_probability=0.5)
    batch = [((1.0, -1.0), 1.0), ((-1.0, 1.0), 0.0)]

    network.learn_batch(0.1, batch)

    assert all(not node.training for node in network.hidden_layers[0].nodes)


def test_randomize_breaks_symmetry_between_nodes_in_the_same_layer():

    network = DropoutBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), drop_probability=0.5)
    assert_randomize_breaks_symmetry(network)


def test_snapshot_and_restore_round_trip():

    network = DropoutBackpropClassifierNetwork.randomized([3, 2], 2, square_bounds(10.0), drop_probability=0.5)
    assert_snapshot_restore_round_trip(network, lambda: network.learn(0.1, (1.0, -2.0), 1.0))


def test_drop_probability_is_a_required_constructor_argument():

    with pytest.raises(TypeError):
        DropoutBackpropClassifierNetwork([4], 2, square_bounds(10.0))  # type: ignore[call-arg]
