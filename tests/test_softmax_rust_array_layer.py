import random

import numpy as np
import pytest

import indrajala_math_rust as pa
from indrajala_ml.model.softmax_output_layer import SoftmaxOutputLayer
from indrajala_ml.model.softmax_rust_array_layer import SoftmaxRustArrayLayer
from indrajala_ml.model.state_layer import StateLayer


def _snapshot_to_softmax_rust_array_layer(softmax_layer: SoftmaxOutputLayer) -> SoftmaxRustArrayLayer:
    rust_layer = SoftmaxRustArrayLayer(softmax_layer.size, len(softmax_layer.input_layer.nodes))
    snapshot = softmax_layer.snapshot_state()
    rust_layer.W = pa.Array([weights for weights, _bias in snapshot])
    rust_layer.b = pa.Array([bias for _weights, bias in snapshot])
    return rust_layer


def test_forward_matches_softmax_output_layer_across_a_random_sweep():

    rng = random.Random(60)
    dimension = 4
    size = 5

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        softmax_layer = SoftmaxOutputLayer(size, state_layer)

        weights = [[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)]
        biases = [rng.uniform(-3.0, 3.0) for _ in range(size)]
        for node, node_weights, bias in zip(softmax_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        rust_layer = _snapshot_to_softmax_rust_array_layer(softmax_layer)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        softmax_layer.forward()
        expected = [node.value() for node in softmax_layer.nodes]

        actual = rust_layer.forward(pa.Array(x))
        assert np.allclose(actual.tolist(), expected, rtol=1e-9, atol=1e-12)
        assert sum(actual.tolist()) == pytest.approx(1.0)


def test_forward_matches_softmax_output_layer_for_large_magnitude_z_without_overflow():

    dimension = 1
    state_layer = StateLayer(dimension, [(-1.0, 1.0)])
    softmax_layer = SoftmaxOutputLayer(3, state_layer)
    for node, (weight, bias) in zip(softmax_layer.nodes, [(1.0, 10_000.0), (1.0, 0.0), (1.0, -10_000.0)]):
        node.update_input_weights([weight])
        node.bias = bias
    state_layer.update_state((0.0,))

    rust_layer = _snapshot_to_softmax_rust_array_layer(softmax_layer)

    softmax_layer.forward()
    expected = [node.value() for node in softmax_layer.nodes]

    actual = rust_layer.forward(pa.Array([0.0])).tolist()
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
    assert actual[0] == pytest.approx(1.0)
    assert actual[1] == pytest.approx(0.0)
    assert actual[2] == pytest.approx(0.0)
    assert not any(v != v for v in actual)  # no NaN


def test_forward_batch_matches_per_row_single_example_results_stacked():

    rng = random.Random(61)
    dimension = 4
    size = 3
    batch_size = 6

    rust_layer = SoftmaxRustArrayLayer(size, dimension)
    rust_layer.W = pa.Array([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    rust_layer.b = pa.Array([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = [[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)]

    expected_rows = [rust_layer.forward(pa.Array(x)).tolist() for x in X]

    actual = rust_layer.forward_batch(pa.Array(X))
    assert np.allclose(actual.tolist(), expected_rows, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_matches_softmax_output_node_across_a_random_sweep():

    rng = random.Random(62)
    dimension = 4
    size = 5

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        softmax_layer = SoftmaxOutputLayer(size, state_layer)

        for node in softmax_layer.nodes:
            node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
            node.bias = rng.uniform(-3.0, 3.0)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        softmax_layer.forward()

        category = rng.randrange(size)
        expected = []
        for i, node in enumerate(softmax_layer.nodes):
            target = 1.0 if i == category else 0.0
            node.compute_output_delta(target)
            expected.append(node.delta)

        rust_layer = _snapshot_to_softmax_rust_array_layer(softmax_layer)
        rust_layer.forward(pa.Array(x))
        reference = [1.0 if i == category else 0.0 for i in range(size)]
        rust_layer.compute_output_delta(pa.Array(reference))

        assert np.allclose(rust_layer.delta.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_batch_matches_per_row_single_example_results_stacked():

    rng = random.Random(63)
    dimension = 4
    size = 3
    batch_size = 5

    rust_layer = SoftmaxRustArrayLayer(size, dimension)
    rust_layer.W = pa.Array([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    rust_layer.b = pa.Array([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = [[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)]
    categories = [rng.randrange(size) for _ in range(batch_size)]
    reference_rows = [[1.0 if i == category else 0.0 for i in range(size)] for category in categories]

    expected_rows = []
    for row in range(batch_size):
        rust_layer.forward(pa.Array(X[row]))
        rust_layer.compute_output_delta(pa.Array(reference_rows[row]))
        expected_rows.append(rust_layer.delta.tolist())

    rust_layer.forward_batch(pa.Array(X))
    rust_layer.compute_output_delta_batch(pa.Array(reference_rows))
    assert np.allclose(rust_layer.delta_batch.tolist(), expected_rows, rtol=1e-9, atol=1e-12)


def test_apply_accumulated_gradient_is_inherited_unchanged_from_rust_array_layer():

    rust_layer = SoftmaxRustArrayLayer(2, 2)
    rust_layer.W = pa.Array([[1.0, 2.0], [3.0, 4.0]])
    rust_layer.b = pa.Array([5.0, 6.0])
    rust_layer.delta = pa.Array([1.0, 1.0])
    rust_layer.accumulate_gradient(pa.Array([1.0, 1.0]))
    rust_layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(rust_layer.W.tolist(), np.array([[0.9, 1.9], [2.9, 3.9]]))
    assert np.allclose(rust_layer.b.tolist(), np.array([4.9, 5.9]))


def test_construction_rejects_a_size_smaller_than_two():

    with pytest.raises(AssertionError):
        SoftmaxRustArrayLayer(1, 3)
