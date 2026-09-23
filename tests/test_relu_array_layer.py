import random

import numpy as np
import pytest

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.relu_array_layer import ReLUArrayLayer
from indrajala_ml.model.relu_layer import ReLULayer
from indrajala_ml.model.state_layer import StateLayer


def _snapshot_to_relu_array_layer(backprop_layer) -> ReLUArrayLayer:
    array_layer = ReLUArrayLayer(backprop_layer.size, len(backprop_layer.input_layer.nodes))
    snapshot = backprop_layer.snapshot_state()
    array_layer.W = np.array([weights for weights, _bias in snapshot])
    array_layer.b = np.array([bias for _weights, bias in snapshot])
    return array_layer


def test_forward_matches_relu_node_across_a_random_sweep_including_the_z_equals_zero_boundary():

    rng = random.Random(41)
    dimension = 5
    size = 4

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        relu_layer = ReLULayer(size, state_layer)

        weights = [[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)]
        biases = [rng.uniform(-3.0, 3.0) for _ in range(size)]
        for node, node_weights, bias in zip(relu_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_layer = _snapshot_to_relu_array_layer(relu_layer)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        relu_layer.forward()
        expected = [node.value() for node in relu_layer.nodes]

        actual = array_layer.forward(np.array(x))
        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_forward_is_exactly_zero_at_the_z_equals_zero_boundary():

    # z == 0 exactly: max(0, 0) == 0, measure-zero in practice but worth pinning explicitly,
    # matching relu_hidden_delta's own either-branch convention for the derivative there
    array_layer = ReLUArrayLayer(1, 1)
    array_layer.W = np.array([[1.0]])
    array_layer.b = np.array([0.0])

    result = array_layer.forward(np.array([0.0]))
    assert result[0] == 0.0


def test_forward_batch_matches_per_row_single_example_results_stacked():

    rng = random.Random(42)
    dimension = 4
    size = 3
    batch_size = 6

    array_layer = ReLUArrayLayer(size, dimension)
    array_layer.W = np.array([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    array_layer.b = np.array([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = np.array([[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)])

    expected_rows = [array_layer.forward(x) for x in X]
    expected = np.stack(expected_rows)

    actual = array_layer.forward_batch(X)
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_compute_hidden_delta_matches_relu_hidden_delta_across_a_random_sweep():

    rng = random.Random(43)
    hidden_size = 5
    next_size = 4

    for _ in range(100):
        state_layer = StateLayer(hidden_size, [(-10.0, 10.0)] * hidden_size)
        hidden_layer = ReLULayer(hidden_size, state_layer)
        next_layer = BackpropLayer(next_size, hidden_layer)

        for node in next_layer.nodes:
            node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(hidden_size)])
            node.bias = rng.uniform(-3.0, 3.0)
            node.delta = rng.uniform(-5.0, 5.0)

        activations = [rng.uniform(-5.0, 5.0) for _ in range(hidden_size)]
        for node, a in zip(hidden_layer.nodes, activations):
            node._activation = a

        expected = []
        for i, node in enumerate(hidden_layer.nodes):
            node.compute_hidden_delta(next_layer.nodes, i)
            expected.append(node.delta)

        array_hidden = ReLUArrayLayer(hidden_size, hidden_size)
        array_hidden.a = np.array(activations)
        array_next = _snapshot_to_relu_array_layer(next_layer)  # W/b only; ArrayLayer-shaped
        array_next.delta = np.array([node.delta for node in next_layer.nodes])

        array_hidden.compute_hidden_delta(array_next)
        assert np.allclose(array_hidden.delta, expected, rtol=1e-9, atol=1e-12)


def test_compute_hidden_delta_is_exactly_zero_at_the_activation_equals_zero_boundary():

    array_hidden = ReLUArrayLayer(1, 0)
    array_hidden.a = np.array([0.0])

    array_next = ReLUArrayLayer(1, 1)
    array_next.W = np.array([[3.0]])
    array_next.delta = np.array([7.0])

    array_hidden.compute_hidden_delta(array_next)
    assert array_hidden.delta[0] == 0.0


def test_compute_hidden_delta_batch_matches_per_row_single_example_results_stacked():

    rng = random.Random(44)
    hidden_size = 5
    next_size = 3
    batch_size = 7

    next_layer = ReLUArrayLayer(next_size, hidden_size)
    next_layer.W = np.array([[rng.uniform(-3.0, 3.0) for _ in range(hidden_size)] for _ in range(next_size)])
    next_layer.delta_batch = np.array(
        [[rng.uniform(-5.0, 5.0) for _ in range(next_size)] for _ in range(batch_size)]
    )

    hidden_layer = ReLUArrayLayer(hidden_size, 0)
    A = np.array([[rng.uniform(-5.0, 5.0) for _ in range(hidden_size)] for _ in range(batch_size)])

    expected_rows = []
    for row_index in range(batch_size):
        hidden_layer.a = A[row_index]
        next_layer.delta = next_layer.delta_batch[row_index]
        hidden_layer.compute_hidden_delta(next_layer)
        expected_rows.append(hidden_layer.delta)
    expected = np.stack(expected_rows)

    hidden_layer.A = A
    hidden_layer.compute_hidden_delta_batch(next_layer)
    assert np.allclose(hidden_layer.delta_batch, expected, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_raises_not_implemented():

    array_layer = ReLUArrayLayer(2, 3)
    with pytest.raises(NotImplementedError):
        array_layer.compute_output_delta(np.array([0.0, 1.0]))


def test_compute_output_delta_batch_raises_not_implemented():

    array_layer = ReLUArrayLayer(2, 3)
    with pytest.raises(NotImplementedError):
        array_layer.compute_output_delta_batch(np.array([[0.0, 1.0]]))


def test_apply_accumulated_gradient_is_inherited_unchanged_from_array_layer():

    # no persistent state, no activation-specific update rule - ReLU only changes the forward/
    # backward formulas, not the weight-update step
    array_layer = ReLUArrayLayer(2, 2)
    array_layer.W = np.array([[1.0, 2.0], [3.0, 4.0]])
    array_layer.b = np.array([5.0, 6.0])
    array_layer.delta = np.array([1.0, 1.0])
    array_layer.accumulate_gradient(np.array([1.0, 1.0]))
    array_layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(array_layer.W, np.array([[0.9, 1.9], [2.9, 3.9]]))
    assert np.allclose(array_layer.b, np.array([4.9, 5.9]))
