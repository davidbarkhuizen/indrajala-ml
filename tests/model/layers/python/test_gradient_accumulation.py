import random

import pytest

from indrajala_ml.model.layers.python.backprop_layer import BackpropLayer
from indrajala_ml.model.layers.python.backprop_node import BackpropNode
from indrajala_ml.model.layers.python.state_layer import StateLayer
from indrajala_ml.model.layers.python.state_node import StateNode
from indrajala_ml.model.optimizers.python_optimizer import PythonOptimizer
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from tests.helpers import LayerOptimizer, WeightSets, approx


def _plain_node(weight: float, bias: float, x: float) -> BackpropNode:
    node = BackpropNode(input_nodes=[StateNode(x)])
    node.update_input_weights([weight])
    node.bias = bias
    return node


def test_accumulate_then_apply_at_batch_size_one_matches_the_direct_formula():

    # batch_size=1 must reproduce weight - learning_rate*delta*x, over a random sweep
    rng = random.Random(0)

    for _ in range(200):
        weight = rng.uniform(-5.0, 5.0)
        bias = rng.uniform(-5.0, 5.0)
        x = rng.uniform(-5.0, 5.0)
        delta = rng.uniform(-5.0, 5.0)
        learning_rate = rng.uniform(0.001, 1.0)

        node = _plain_node(weight, bias, x)
        node.delta = delta
        node.accumulate_gradient()
        LayerOptimizer(WeightSets(node)).apply(learning_rate, batch_size=1)

        assert node.input_node_weights[0] == approx(weight - learning_rate * delta * x)
        assert node.bias == approx(bias - learning_rate * delta)


def test_accumulate_gradient_sums_across_multiple_examples_before_any_weight_write():

    # two examples with different x: no weight moves until the optimizer applies them
    x_node = StateNode(1.0)
    node = BackpropNode(input_nodes=[x_node])
    node.update_input_weights([0.5])
    node.bias = 0.1

    x_node.update_value(1.0)
    node.delta = 0.2
    node.accumulate_gradient()
    assert node.input_node_weights[0] == approx(0.5)  # untouched
    assert node.bias == approx(0.1)  # untouched

    x_node.update_value(2.0)
    node.delta = -0.1
    node.accumulate_gradient()

    # accum_w = 0.2*1.0 + (-0.1)*2.0 = 0.0, accum_b = 0.2 + (-0.1) = 0.1
    LayerOptimizer(WeightSets(node)).apply(0.1, batch_size=2)
    assert node.input_node_weights[0] == approx(0.5 - 0.1 * (0.0 / 2))
    assert node.bias == approx(0.1 - 0.1 * (0.1 / 2))


def test_apply_resets_the_accumulator():

    x_node = StateNode(1.0)
    node = BackpropNode(input_nodes=[x_node])
    node.update_input_weights([0.5])
    node.bias = 0.1

    optimizer = LayerOptimizer(WeightSets(node))
    node.delta = 0.2
    node.accumulate_gradient()
    optimizer.apply(0.1, batch_size=1)

    weight_after_first_apply = node.input_node_weights[0]
    bias_after_first_apply = node.bias

    # a second apply with nothing accumulated in between must be a no-op (accumulator reset to
    # zero, not left over from the batch just applied)
    optimizer.apply(0.1, batch_size=1)
    assert node.input_node_weights[0] == approx(weight_after_first_apply)
    assert node.bias == approx(bias_after_first_apply)


def _batch_examples() -> list[tuple[float, float]]:
    # (x, delta) pairs - shared by the momentum/L2 batch tests below so both exercise the same
    # accumulated gradient (accum_w = 0.6, accum_b = 0.6, averaged over batch_size=2 -> 0.3 each)
    return [(1.0, 0.2), (1.0, 0.4)]


def _accumulate_batch(node: BackpropNode, examples: list[tuple[float, float]]) -> None:
    (x_node,) = node.input_nodes
    assert isinstance(x_node, StateNode)
    for x, delta in examples:
        x_node.update_value(x)
        node.delta = delta
        node.accumulate_gradient()


def test_momentum_apply_matches_hand_computed_batch_values():

    # hand-derived, learning_rate=0.1, momentum=0.9, batch_size=2, _batch_examples() (averaged
    # gradient 0.3 for weight and bias):
    #   batch 1: delta_w = 0.1*0.3 + 0.9*0.0 = 0.03 -> weight = 0.5-0.03 = 0.47
    #            bias_delta = 0.1*0.3 + 0.9*0.0 = 0.03 -> bias = 0.1-0.03 = 0.07
    #   batch 2 (same examples again): delta_w = 0.1*0.3 + 0.9*0.03 = 0.057 -> weight = 0.413
    #            bias_delta = 0.1*0.3 + 0.9*0.03 = 0.057 -> bias = 0.013
    node = _plain_node(0.5, 0.1, 0.0)
    optimizer = LayerOptimizer(WeightSets(node), Momentum(0.9))

    _accumulate_batch(node, _batch_examples())
    optimizer.apply(0.1, batch_size=2)
    assert node.input_node_weights[0] == approx(0.47)
    assert node.bias == approx(0.07)

    _accumulate_batch(node, _batch_examples())
    optimizer.apply(0.1, batch_size=2)
    assert node.input_node_weights[0] == approx(0.413)
    assert node.bias == approx(0.013)


def test_l2_apply_matches_hand_computed_batch_values():

    # hand-derived, learning_rate=0.1, l2_lambda=0.1, the same averaged gradient 0.3:
    #   weight = 0.5 - 0.1*(0.3 + 0.1*0.5) = 0.5 - 0.1*0.35 = 0.465
    #   bias = 0.1 - 0.1*0.3 = 0.07 (never regularized)
    node = _plain_node(0.5, 0.1, 0.0)

    _accumulate_batch(node, _batch_examples())
    LayerOptimizer(WeightSets(node), WeightDecay(0.1)).apply(0.1, batch_size=2)
    assert node.input_node_weights[0] == approx(0.465)
    assert node.bias == approx(0.07)


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), WeightDecay(0.1), Adam()], ids=type)
def test_step_single_is_bit_identical_to_accumulate_then_apply_at_batch_size_one(rule: UpdateRule):

    # the single-example path (learn) and a one-example batch (learn_batch) through every rule
    rng = random.Random(repr(rule))
    for _ in range(50):
        x = rng.uniform(-5.0, 5.0)
        weight = rng.uniform(-5.0, 5.0)
        bias = rng.uniform(-5.0, 5.0)
        delta = rng.uniform(-5.0, 5.0)
        learning_rate = rng.uniform(0.001, 1.0)

        layers: list[BackpropLayer] = []
        for _path in range(2):
            input_layer = StateLayer(1, [(-5.0, 5.0)])
            input_layer.update_state((x,))
            layer = BackpropLayer(size=1, input_layer=input_layer)
            layer.nodes[0].update_input_weights([weight])
            layer.nodes[0].bias = bias
            layer.nodes[0].delta = delta
            layers.append(layer)
        via_step_single, via_split = layers

        optimizer = PythonOptimizer(rule)
        optimizer.begin_step()
        optimizer.step_single(0, via_step_single, learning_rate)

        optimizer = PythonOptimizer(rule)
        optimizer.begin_step()
        via_split.accumulate_gradients()
        optimizer.apply(0, via_split, learning_rate, 1)

        assert via_step_single.nodes[0].input_node_weights[0] == via_split.nodes[0].input_node_weights[0]
        assert via_step_single.nodes[0].bias == via_split.nodes[0].bias
