import random

import numpy as np
import pytest

import indrajala_math_rust as pa
from indrajala_ml.model.relu_layer import ReLULayer
from indrajala_ml.model.relu_rust_array_layer import ReLURustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.state_layer import StateLayer


def _snapshot_to_relu_rust_array_layer(backprop_layer) -> ReLURustArrayLayer:
    rust_layer = ReLURustArrayLayer(backprop_layer.size, len(backprop_layer.input_layer.nodes))
    snapshot = backprop_layer.snapshot_state()
    rust_layer.W = pa.Array([weights for weights, _bias in snapshot])
    rust_layer.b = pa.Array([bias for _weights, bias in snapshot])
    return rust_layer


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

        rust_layer = _snapshot_to_relu_rust_array_layer(relu_layer)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        relu_layer.forward()
        expected = [node.value() for node in relu_layer.nodes]

        actual = rust_layer.forward(pa.Array(x))
        assert np.allclose(actual.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_forward_batch_matches_per_row_single_example_results_stacked():

    rng = random.Random(42)
    dimension = 4
    size = 3
    batch_size = 6

    rust_layer = ReLURustArrayLayer(size, dimension)
    rust_layer.W = pa.Array([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    rust_layer.b = pa.Array([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = [[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)]

    expected_rows = [rust_layer.forward(pa.Array(x)).tolist() for x in X]

    actual = rust_layer.forward_batch(pa.Array(X))
    assert np.allclose(actual.tolist(), expected_rows, rtol=1e-9, atol=1e-12)


def test_compute_hidden_delta_matches_relu_hidden_delta_across_a_random_sweep():

    rng = random.Random(43)
    hidden_size = 5
    next_size = 4

    rust_hidden = ReLURustArrayLayer(hidden_size, hidden_size)
    a_data = [rng.uniform(-5.0, 5.0) for _ in range(hidden_size)]
    rust_hidden.a = pa.Array(a_data)

    rust_next = RustArrayLayer(next_size, hidden_size)
    rust_next.W = pa.Array([[rng.uniform(-3.0, 3.0) for _ in range(hidden_size)] for _ in range(next_size)])
    rust_next.delta = pa.Array([rng.uniform(-5.0, 5.0) for _ in range(next_size)])

    rust_hidden.compute_hidden_delta(rust_next)

    downstream = np.array(rust_next.W.tolist()).T @ np.array(rust_next.delta.tolist())
    expected = downstream * (np.array(a_data) > 0.0)
    assert np.allclose(rust_hidden.delta.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_raises_not_implemented():

    rust_layer = ReLURustArrayLayer(2, 3)
    with pytest.raises(NotImplementedError):
        rust_layer.compute_output_delta(pa.Array([0.0, 1.0]))


def test_compute_output_delta_batch_raises_not_implemented():

    rust_layer = ReLURustArrayLayer(2, 3)
    with pytest.raises(NotImplementedError):
        rust_layer.compute_output_delta_batch(pa.Array([[0.0, 1.0]]))


def test_apply_accumulated_gradient_is_inherited_unchanged_from_rust_array_layer():

    rust_layer = ReLURustArrayLayer(2, 2)
    rust_layer.W = pa.Array([[1.0, 2.0], [3.0, 4.0]])
    rust_layer.b = pa.Array([5.0, 6.0])
    rust_layer.delta = pa.Array([1.0, 1.0])
    rust_layer.accumulate_gradient(pa.Array([1.0, 1.0]))
    rust_layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(rust_layer.W.tolist(), np.array([[0.9, 1.9], [2.9, 3.9]]))
    assert np.allclose(rust_layer.b.tolist(), np.array([4.9, 5.9]))
