import random

import pytest

from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.l2_regularization_layer import make_l2_node_cls
from indrajala_ml.model.momentum_layer import make_momentum_node_cls
from indrajala_ml.model.state_node import StateNode


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
        node.apply_accumulated_gradient(learning_rate, batch_size=1)

        assert node.input_node_weights[0] == pytest.approx(weight - learning_rate * delta * x)
        assert node.bias == pytest.approx(bias - learning_rate * delta)


def test_accumulate_then_apply_at_batch_size_one_is_bit_identical_to_apply_gradient():

    rng = random.Random(1)

    for _ in range(200):
        weight = rng.uniform(-5.0, 5.0)
        bias = rng.uniform(-5.0, 5.0)
        x = rng.uniform(-5.0, 5.0)
        delta = rng.uniform(-5.0, 5.0)
        learning_rate = rng.uniform(0.001, 1.0)

        via_split = _plain_node(weight, bias, x)
        via_split.delta = delta
        via_split.accumulate_gradient()
        via_split.apply_accumulated_gradient(learning_rate, batch_size=1)

        via_apply_gradient = _plain_node(weight, bias, x)
        via_apply_gradient.delta = delta
        via_apply_gradient.apply_gradient(learning_rate)

        assert via_split.input_node_weights[0] == via_apply_gradient.input_node_weights[0]
        assert via_split.bias == via_apply_gradient.bias


def test_accumulate_gradient_sums_across_multiple_examples_before_any_weight_write():

    # two examples with different x: no weight moves until apply_accumulated_gradient
    x_node = StateNode(1.0)
    node = BackpropNode(input_nodes=[x_node])
    node.update_input_weights([0.5])
    node.bias = 0.1

    x_node.update_value(1.0)
    node.delta = 0.2
    node.accumulate_gradient()
    assert node.input_node_weights[0] == pytest.approx(0.5)  # untouched
    assert node.bias == pytest.approx(0.1)  # untouched

    x_node.update_value(2.0)
    node.delta = -0.1
    node.accumulate_gradient()

    # accum_w = 0.2*1.0 + (-0.1)*2.0 = 0.0, accum_b = 0.2 + (-0.1) = 0.1
    node.apply_accumulated_gradient(0.1, batch_size=2)
    assert node.input_node_weights[0] == pytest.approx(0.5 - 0.1 * (0.0 / 2))
    assert node.bias == pytest.approx(0.1 - 0.1 * (0.1 / 2))


def test_apply_accumulated_gradient_resets_the_accumulator():

    x_node = StateNode(1.0)
    node = BackpropNode(input_nodes=[x_node])
    node.update_input_weights([0.5])
    node.bias = 0.1

    node.delta = 0.2
    node.accumulate_gradient()
    node.apply_accumulated_gradient(0.1, batch_size=1)

    weight_after_first_apply = node.input_node_weights[0]
    bias_after_first_apply = node.bias

    # a second apply with nothing accumulated in between must be a no-op (accumulator reset to
    # zero, not left over from the batch just applied)
    node.apply_accumulated_gradient(0.1, batch_size=1)
    assert node.input_node_weights[0] == pytest.approx(weight_after_first_apply)
    assert node.bias == pytest.approx(bias_after_first_apply)


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


def test_momentum_apply_accumulated_gradient_matches_hand_computed_batch_values():

    # hand-derived, learning_rate=0.1, momentum=0.9, batch_size=2, _batch_examples() (averaged
    # gradient 0.3 for weight and bias):
    #   batch 1: delta_w = 0.1*0.3 + 0.9*0.0 = 0.03 -> weight = 0.5-0.03 = 0.47
    #            bias_delta = 0.1*0.3 + 0.9*0.0 = 0.03 -> bias = 0.1-0.03 = 0.07
    #   batch 2 (same examples again): delta_w = 0.1*0.3 + 0.9*0.03 = 0.057 -> weight = 0.413
    #            bias_delta = 0.1*0.3 + 0.9*0.03 = 0.057 -> bias = 0.013
    node_cls = make_momentum_node_cls(0.9)
    node = node_cls(input_nodes=[StateNode(0.0)])
    node.update_input_weights([0.5])
    node.bias = 0.1

    _accumulate_batch(node, _batch_examples())
    node.apply_accumulated_gradient(0.1, batch_size=2)
    assert node.input_node_weights[0] == pytest.approx(0.47)
    assert node.bias == pytest.approx(0.07)

    _accumulate_batch(node, _batch_examples())
    node.apply_accumulated_gradient(0.1, batch_size=2)
    assert node.input_node_weights[0] == pytest.approx(0.413)
    assert node.bias == pytest.approx(0.013)


def test_l2_apply_accumulated_gradient_matches_hand_computed_batch_values():

    # hand-derived, learning_rate=0.1, l2_lambda=0.1, the same averaged gradient 0.3:
    #   weight = 0.5 - 0.1*(0.3 + 0.1*0.5) = 0.5 - 0.1*0.35 = 0.465
    #   bias = 0.1 - 0.1*0.3 = 0.07 (never regularized)
    node_cls = make_l2_node_cls(0.1)
    node = node_cls(input_nodes=[StateNode(0.0)])
    node.update_input_weights([0.5])
    node.bias = 0.1

    _accumulate_batch(node, _batch_examples())
    node.apply_accumulated_gradient(0.1, batch_size=2)
    assert node.input_node_weights[0] == pytest.approx(0.465)
    assert node.bias == pytest.approx(0.07)


def test_momentum_and_l2_batch_size_one_still_match_apply_gradient_exactly():

    # the batch_size=1 check for the two subclasses that override apply_accumulated_gradient
    for node_cls in (make_momentum_node_cls(0.9), make_l2_node_cls(0.1)):
        rng = random.Random(hash(node_cls.__name__) & 0xFFFF)
        for _ in range(50):
            weight = rng.uniform(-5.0, 5.0)
            bias = rng.uniform(-5.0, 5.0)
            x = rng.uniform(-5.0, 5.0)
            delta = rng.uniform(-5.0, 5.0)
            learning_rate = rng.uniform(0.001, 1.0)

            via_split = node_cls(input_nodes=[StateNode(x)])
            via_split.update_input_weights([weight])
            via_split.bias = bias
            via_split.delta = delta
            via_split.accumulate_gradient()
            via_split.apply_accumulated_gradient(learning_rate, batch_size=1)

            via_apply_gradient = node_cls(input_nodes=[StateNode(x)])
            via_apply_gradient.update_input_weights([weight])
            via_apply_gradient.bias = bias
            via_apply_gradient.delta = delta
            via_apply_gradient.apply_gradient(learning_rate)

            assert via_split.input_node_weights[0] == via_apply_gradient.input_node_weights[0]
            assert via_split.bias == via_apply_gradient.bias
