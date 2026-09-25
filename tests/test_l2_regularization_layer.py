import pytest

from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.l2_regularization_layer import make_l2_layer_cls, make_l2_node_cls
from indrajala_ml.model.state_layer import StateLayer
from indrajala_ml.model.state_node import StateNode


def _l2_node(l2_lambda: float, weight: float, bias: float):
    node_cls = make_l2_node_cls(l2_lambda)
    x = StateNode(1.0)
    node = node_cls(input_nodes=[x])
    node.update_input_weights([weight])
    node.bias = bias
    return node


def test_apply_gradient_adds_the_l2_penalty_to_the_weight_by_hand():

    # hand-derived, x=1.0, delta=0.2, learning_rate=0.1, l2_lambda=0.1:
    #   new_weight = 0.5 - 0.1*(0.2*1.0 + 0.1*0.5) = 0.5 - 0.1*0.25 = 0.475
    #   new_bias = 0.1 - 0.1*0.2 = 0.08 (biases aren't regularized)
    node = _l2_node(0.1, weight=0.5, bias=0.1)

    node.delta = 0.2
    node.apply_gradient(0.1)

    assert node.input_node_weights[0] == pytest.approx(0.475)
    assert node.bias == pytest.approx(0.08)


def test_bias_is_never_regularized():

    small_l2 = _l2_node(0.001, weight=0.5, bias=0.1)
    large_l2 = _l2_node(50.0, weight=0.5, bias=0.1)

    small_l2.delta = 0.2
    large_l2.delta = 0.2
    small_l2.apply_gradient(0.1)
    large_l2.apply_gradient(0.1)

    assert small_l2.bias == pytest.approx(0.08)
    assert large_l2.bias == pytest.approx(0.08)
    assert small_l2.bias == large_l2.bias
    # but the weights must differ, since the penalty *does* apply there
    assert small_l2.input_node_weights[0] != large_l2.input_node_weights[0]


def test_l2_lambda_zero_matches_plain_sgd_exactly():

    l2_node = _l2_node(0.0, weight=0.5, bias=0.1)
    plain_node = BackpropNode(input_nodes=[StateNode(1.0)])
    plain_node.update_input_weights([0.5])
    plain_node.bias = 0.1

    l2_node.delta = 0.2
    plain_node.delta = 0.2
    l2_node.apply_gradient(0.1)
    plain_node.apply_gradient(0.1)

    assert l2_node.input_node_weights[0] == pytest.approx(plain_node.input_node_weights[0])
    assert l2_node.bias == pytest.approx(plain_node.bias)


def test_a_large_enough_weight_shrinks_even_with_zero_delta():

    # weight decay: with delta=0, the l2_lambda*weight term alone shrinks the weight
    node = _l2_node(0.5, weight=10.0, bias=0.0)

    node.delta = 0.0
    node.apply_gradient(0.1)

    # new_weight = 10.0 - 0.1*(0.0 + 0.5*10.0) = 10.0 - 0.5 = 9.5
    assert node.input_node_weights[0] == pytest.approx(9.5)
    assert node.bias == 0.0


def test_make_l2_layer_cls_builds_nodes_of_the_configured_l2_node_class():

    input_layer = StateLayer(2, [(-10.0, 10.0), (-10.0, 10.0)])
    layer_cls = make_l2_layer_cls(0.1)
    layer = layer_cls(size=3, input_layer=input_layer)

    node_cls = make_l2_node_cls(0.1)
    assert all(type(node).__name__ == node_cls.__name__ for node in layer.nodes)
