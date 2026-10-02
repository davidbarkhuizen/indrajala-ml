from collections.abc import Callable

from indrajala_ml.model.layers.python.backprop_node import BackpropNode
from indrajala_ml.model.layers.python.state_node import StateNode
from indrajala_ml.model.specs.update_rules import SGD, UpdateRule, WeightDecay
from tests.helpers import LayerOptimizer, WeightSets, approx


def _node(rule: UpdateRule, weight: float, bias: float) -> tuple[BackpropNode, Callable[[float], None]]:
    # a one-input node reading x=1.0, and its single-example step under rule
    node = BackpropNode(input_nodes=[StateNode(1.0)])
    node.update_input_weights([weight])
    node.bias = bias
    return node, LayerOptimizer(WeightSets(node), rule).apply_single(node)


def test_a_step_adds_the_l2_penalty_to_the_weight_by_hand():

    # hand-derived, x=1.0, delta=0.2, learning_rate=0.1, l2_lambda=0.1:
    #   new_weight = 0.5 - 0.1*(0.2*1.0 + 0.1*0.5) = 0.5 - 0.1*0.25 = 0.475
    #   new_bias = 0.1 - 0.1*0.2 = 0.08 (biases aren't regularized)
    node, step = _node(WeightDecay(0.1), weight=0.5, bias=0.1)

    node.delta = 0.2
    step(0.1)

    assert node.input_node_weights[0] == approx(0.475)
    assert node.bias == approx(0.08)


def test_bias_is_never_regularized():

    small_l2, small_step = _node(WeightDecay(0.001), weight=0.5, bias=0.1)
    large_l2, large_step = _node(WeightDecay(50.0), weight=0.5, bias=0.1)

    small_l2.delta = 0.2
    large_l2.delta = 0.2
    small_step(0.1)
    large_step(0.1)

    assert small_l2.bias == approx(0.08)
    assert large_l2.bias == approx(0.08)
    assert small_l2.bias == large_l2.bias
    # but the weights must differ, since the penalty *does* apply there
    assert small_l2.input_node_weights[0] != large_l2.input_node_weights[0]


def test_l2_lambda_zero_matches_plain_sgd_exactly():

    l2_node, l2_step = _node(WeightDecay(0.0), weight=0.5, bias=0.1)
    plain_node, plain_step = _node(SGD(), weight=0.5, bias=0.1)

    l2_node.delta = 0.2
    plain_node.delta = 0.2
    l2_step(0.1)
    plain_step(0.1)

    assert l2_node.input_node_weights[0] == approx(plain_node.input_node_weights[0])
    assert l2_node.bias == approx(plain_node.bias)


def test_a_large_enough_weight_shrinks_even_with_zero_delta():

    # weight decay: with delta=0, the l2_lambda*weight term alone shrinks the weight
    node, step = _node(WeightDecay(0.5), weight=10.0, bias=0.0)

    node.delta = 0.0
    step(0.1)

    # new_weight = 10.0 - 0.1*(0.0 + 0.5*10.0) = 10.0 - 0.5 = 9.5
    assert node.input_node_weights[0] == approx(9.5)
    assert node.bias == 0.0
