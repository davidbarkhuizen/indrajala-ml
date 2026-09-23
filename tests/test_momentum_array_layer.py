import random

import numpy as np

from indrajala_ml.model.momentum_array_layer import MomentumArrayLayer
from indrajala_ml.model.momentum_layer import make_momentum_layer_cls
from indrajala_ml.model.state_layer import StateLayer

MOMENTUM = 0.5


def _snapshot_to_momentum_array_layer(backprop_layer) -> MomentumArrayLayer:
    array_layer = MomentumArrayLayer(backprop_layer.size, len(backprop_layer.input_layer.nodes), MOMENTUM)
    snapshot = backprop_layer.snapshot_state()
    array_layer.W = np.array([weights for weights, _bias in snapshot])
    array_layer.b = np.array([bias for _weights, bias in snapshot])
    return array_layer


def test_accumulate_then_apply_at_batch_size_one_matches_momentum_backprop_node_at_every_step():

    # mirrors test_adam_array_layer.py's own equivalent test - momentum's own previous-delta
    # state only actually exercises its accumulation logic across repeated steps, unlike a
    # stateless update where a single comparison would do
    rng = random.Random(31)
    dimension = 5
    size = 4
    momentum_layer_cls = make_momentum_layer_cls(MOMENTUM)

    state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
    backprop_layer = momentum_layer_cls(size=size, input_layer=state_layer)
    for node in backprop_layer.nodes:
        node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
        node.bias = rng.uniform(-3.0, 3.0)

    array_layer = _snapshot_to_momentum_array_layer(backprop_layer)
    learning_rate = rng.uniform(0.001, 1.0)

    for _ in range(10):
        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        deltas = [rng.uniform(-5.0, 5.0) for _ in range(size)]

        for node, delta in zip(backprop_layer.nodes, deltas):
            node.delta = delta
            node.accumulate_gradient()
            node.apply_accumulated_gradient(learning_rate, batch_size=1)

        array_layer.delta = np.array(deltas)
        array_layer.accumulate_gradient(np.array(x))
        array_layer.apply_accumulated_gradient(learning_rate, batch_size=1)

        expected_W = np.array([node.input_node_weights for node in backprop_layer.nodes])
        expected_b = np.array([node.bias for node in backprop_layer.nodes])
        assert np.allclose(array_layer.W, expected_W, rtol=1e-9, atol=1e-12)
        assert np.allclose(array_layer.b, expected_b, rtol=1e-9, atol=1e-12)


def test_accumulate_across_a_batch_then_apply_matches_momentum_backprop_node_at_every_batch():

    rng = random.Random(32)
    dimension = 4
    size = 3
    batch_size = 6
    momentum_layer_cls = make_momentum_layer_cls(MOMENTUM)

    state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
    backprop_layer = momentum_layer_cls(size=size, input_layer=state_layer)
    for node in backprop_layer.nodes:
        node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
        node.bias = rng.uniform(-3.0, 3.0)

    array_layer = _snapshot_to_momentum_array_layer(backprop_layer)
    learning_rate = rng.uniform(0.001, 1.0)

    for _ in range(5):
        examples = [
            ([rng.uniform(-10.0, 10.0) for _ in range(dimension)], [rng.uniform(-5.0, 5.0) for _ in range(size)])
            for _ in range(batch_size)
        ]

        for x, deltas in examples:
            state_layer.update_state(tuple(x))
            for node, delta in zip(backprop_layer.nodes, deltas):
                node.delta = delta
                node.accumulate_gradient()

            array_layer.delta = np.array(deltas)
            array_layer.accumulate_gradient(np.array(x))

        for node in backprop_layer.nodes:
            node.apply_accumulated_gradient(learning_rate, batch_size)
        array_layer.apply_accumulated_gradient(learning_rate, batch_size)

        expected_W = np.array([node.input_node_weights for node in backprop_layer.nodes])
        expected_b = np.array([node.bias for node in backprop_layer.nodes])
        assert np.allclose(array_layer.W, expected_W, rtol=1e-9, atol=1e-12)
        assert np.allclose(array_layer.b, expected_b, rtol=1e-9, atol=1e-12)


def test_apply_accumulated_gradient_resets_the_accumulator():

    array_layer = MomentumArrayLayer(3, 2, MOMENTUM)
    array_layer.delta = np.array([0.1, 0.2, 0.3])
    array_layer.accumulate_gradient(np.array([1.0, 2.0]))
    array_layer.apply_accumulated_gradient(0.1, batch_size=1)

    assert np.allclose(array_layer._grad_W, np.zeros((3, 2)))
    assert np.allclose(array_layer._grad_b, np.zeros(3))


def test_previous_delta_carries_zero_on_the_first_step():

    # with momentum's own previous-delta state zero-initialized, the first apply call should be
    # numerically identical to plain SGD (the momentum term contributes nothing yet)
    array_layer = MomentumArrayLayer(2, 2, momentum=0.9)
    array_layer.W = np.array([[1.0, 2.0], [3.0, 4.0]])
    array_layer.b = np.array([5.0, 6.0])
    array_layer._grad_W = np.array([[1.0, 1.0], [1.0, 1.0]])
    array_layer._grad_b = np.array([1.0, 1.0])

    array_layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(array_layer.W, np.array([[0.9, 1.9], [2.9, 3.9]]))
    assert np.allclose(array_layer.b, np.array([4.9, 5.9]))
