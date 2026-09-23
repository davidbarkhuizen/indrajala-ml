import random

import numpy as np

from indrajala_ml.model.l2_array_layer import L2ArrayLayer
from indrajala_ml.model.l2_regularization_layer import make_l2_layer_cls
from indrajala_ml.model.state_layer import StateLayer

L2_LAMBDA = 0.05


def _snapshot_to_l2_array_layer(backprop_layer) -> L2ArrayLayer:
    array_layer = L2ArrayLayer(backprop_layer.size, len(backprop_layer.input_layer.nodes), L2_LAMBDA)
    snapshot = backprop_layer.snapshot_state()
    array_layer.W = np.array([weights for weights, _bias in snapshot])
    array_layer.b = np.array([bias for _weights, bias in snapshot])
    return array_layer


def test_accumulate_then_apply_at_batch_size_one_matches_l2_backprop_node_at_every_step():

    # mirrors test_adam_array_layer.py's own
    # test_accumulate_then_apply_at_batch_size_one_matches_adam_backprop_node_at_every_step -
    # L2 needs no persistent state between steps, but checking after every one of several steps
    # (not just once) still catches a mistake that only shows up once W itself has moved away
    # from its starting value (the penalty term is a function of the *current* weight).
    rng = random.Random(21)
    dimension = 5
    size = 4
    l2_layer_cls = make_l2_layer_cls(L2_LAMBDA)

    state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
    backprop_layer = l2_layer_cls(size=size, input_layer=state_layer)
    for node in backprop_layer.nodes:
        node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
        node.bias = rng.uniform(-3.0, 3.0)

    array_layer = _snapshot_to_l2_array_layer(backprop_layer)
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


def test_accumulate_across_a_batch_then_apply_matches_l2_backprop_node_at_every_batch():

    rng = random.Random(22)
    dimension = 4
    size = 3
    batch_size = 6
    l2_layer_cls = make_l2_layer_cls(L2_LAMBDA)

    state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
    backprop_layer = l2_layer_cls(size=size, input_layer=state_layer)
    for node in backprop_layer.nodes:
        node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
        node.bias = rng.uniform(-3.0, 3.0)

    array_layer = _snapshot_to_l2_array_layer(backprop_layer)
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

    array_layer = L2ArrayLayer(3, 2, L2_LAMBDA)
    array_layer.delta = np.array([0.1, 0.2, 0.3])
    array_layer.accumulate_gradient(np.array([1.0, 2.0]))
    array_layer.apply_accumulated_gradient(0.1, batch_size=1)

    assert np.allclose(array_layer._grad_W, np.zeros((3, 2)))
    assert np.allclose(array_layer._grad_b, np.zeros(3))


def test_bias_is_never_regularized():

    # l2_lambda should only ever appear in the W update - a nonzero bias gradient with a zeroed
    # weight gradient should move the bias exactly as much as the base ArrayLayer would (no l2
    # term at all), confirming the penalty is W-only, matching make_l2_node_cls's own comment.
    array_layer = L2ArrayLayer(2, 2, l2_lambda=0.5)
    array_layer.W = np.array([[1.0, 2.0], [3.0, 4.0]])
    array_layer.b = np.array([5.0, 6.0])
    array_layer.delta = np.array([0.0, 0.0])
    array_layer.accumulate_gradient(np.array([0.0, 0.0]))
    array_layer._grad_b = np.array([2.0, 4.0])

    array_layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(array_layer.b, np.array([5.0 - 0.1 * 2.0, 6.0 - 0.1 * 4.0]))
