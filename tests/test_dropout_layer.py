from unittest.mock import patch

import pytest

from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.dropout_layer import make_dropout_layer_cls
from indrajala_ml.model.state_layer import StateLayer

# the same fixture point every other activation-function/sibling test in this codebase uses
# (test_backprop_model.py, test_relu_layer.py, test_momentum_backprop_model.py, ...): weight=0.5,
# bias=0.1, x=2.0 -> z=1.1, a_h=sigmoid(1.1)=0.7502601055951177 - directly comparable across
# every sibling's own fixture, not re-derived here
Z = 1.1
BASE_ACTIVATION = 0.7502601055951177


def _dropout_node(drop_probability: float, weight: float = 0.5, bias: float = 0.1, x: float = 2.0):
    input_layer = StateLayer(1, [(-10.0, 10.0)])
    input_layer.update_state((x,))
    layer_cls = make_dropout_layer_cls(drop_probability)
    layer = layer_cls(size=1, input_layer=input_layer)
    node = layer.nodes[0]
    node.update_input_weights([weight])
    node.bias = bias
    return layer, node


def test_forward_at_eval_mode_matches_a_plain_sigmoid_no_rescale():

    # training defaults to False - the safe default if set_training_mode is never called
    _, node = _dropout_node(drop_probability=0.5)

    node.forward()

    assert node.value() == pytest.approx(BASE_ACTIVATION)


def test_forward_in_training_mode_when_kept_rescales_by_one_over_keep_probability():

    layer, node = _dropout_node(drop_probability=0.5)
    layer.set_training_mode(True)

    with patch("random.random", return_value=0.9):  # 0.9 >= 0.5 -> kept
        node.forward()

    assert node.value() == pytest.approx(BASE_ACTIVATION / 0.5)


def test_forward_in_training_mode_when_dropped_is_exactly_zero():

    layer, node = _dropout_node(drop_probability=0.5)
    layer.set_training_mode(True)

    with patch("random.random", return_value=0.1):  # 0.1 < 0.5 -> dropped
        node.forward()

    assert node.value() == 0.0


def test_set_training_mode_false_reverts_to_eval_behavior():

    layer, node = _dropout_node(drop_probability=0.5)
    layer.set_training_mode(True)
    with patch("random.random", return_value=0.1):
        node.forward()
    assert node.value() == 0.0  # dropped, while still in training mode

    layer.set_training_mode(False)
    node.forward()

    assert node.value() == pytest.approx(BASE_ACTIVATION)


def test_compute_hidden_delta_when_kept_uses_the_unscaled_sigmoid_derivative():

    # the derivative factor must be base*(1-base) (the *pre*-scaling activation), not
    # value()*(1-value()) the way
    # BackpropNode.compute_hidden_delta's own a*(1-a) would - see the hand-derivation in
    # dropout_layer.py's own compute_hidden_delta docstring
    layer, node = _dropout_node(drop_probability=0.5)
    layer.set_training_mode(True)
    with patch("random.random", return_value=0.9):  # kept
        node.forward()

    next_node = BackpropNode(input_nodes=[node])
    next_node.update_input_weights([0.8])
    next_node.delta = -0.5

    node.compute_hidden_delta([next_node], own_index=0)

    sigmoid_derivative = BASE_ACTIVATION * (1.0 - BASE_ACTIVATION)
    expected = (-0.5 * 0.8) * sigmoid_derivative / 0.5
    assert node.delta == pytest.approx(expected)


def test_compute_hidden_delta_is_zero_when_the_unit_was_dropped():

    # matches ReLUNode's own dead-unit precedent: zero delta, zero gradient, incoming weights
    # untouched this step, regardless of how large the downstream error is
    layer, node = _dropout_node(drop_probability=0.5)
    layer.set_training_mode(True)
    with patch("random.random", return_value=0.1):  # dropped
        node.forward()

    next_node = BackpropNode(input_nodes=[node])
    next_node.update_input_weights([0.8])
    next_node.delta = -0.5

    node.compute_hidden_delta([next_node], own_index=0)

    assert node.delta == 0.0


def test_compute_hidden_delta_at_eval_mode_uses_no_rescale():

    _, node = _dropout_node(drop_probability=0.5)
    node.forward()  # training defaults to False

    next_node = BackpropNode(input_nodes=[node])
    next_node.update_input_weights([0.8])
    next_node.delta = -0.5

    node.compute_hidden_delta([next_node], own_index=0)

    sigmoid_derivative = BASE_ACTIVATION * (1.0 - BASE_ACTIVATION)
    expected = (-0.5 * 0.8) * sigmoid_derivative
    assert node.delta == pytest.approx(expected)


def test_compute_output_delta_raises_since_dropout_is_hidden_layer_only():

    _, node = _dropout_node(drop_probability=0.5)
    node.forward()

    with pytest.raises(NotImplementedError):
        node.compute_output_delta(1.0)


def test_drop_probability_of_zero_never_drops():

    layer, node = _dropout_node(drop_probability=0.0)
    layer.set_training_mode(True)

    with patch("random.random", return_value=0.0):  # 0.0 >= 0.0 -> kept
        node.forward()

    assert node.value() == pytest.approx(BASE_ACTIVATION)  # keep_probability=1.0, no rescale


def test_drop_probability_of_one_is_rejected():

    with pytest.raises(AssertionError):
        make_dropout_layer_cls(drop_probability=1.0)
