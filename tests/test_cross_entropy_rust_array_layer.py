import random

import numpy as np

import indrajala_math_rust as pa
from indrajala_ml.model.binary_cross_entropy_backprop_classifier_network import CrossEntropyOutputLayer
from indrajala_ml.model.cross_entropy_rust_array_layer import CrossEntropyRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.state_layer import StateLayer


def _snapshot_to_cross_entropy_rust_array_layer(cross_entropy_layer: CrossEntropyOutputLayer) -> CrossEntropyRustArrayLayer:
    rust_layer = CrossEntropyRustArrayLayer(cross_entropy_layer.size, len(cross_entropy_layer.input_layer.nodes))
    snapshot = cross_entropy_layer.snapshot_state()
    rust_layer.W = pa.Array([weights for weights, _bias in snapshot])
    rust_layer.b = pa.Array([bias for _weights, bias in snapshot])
    return rust_layer


def test_compute_output_delta_matches_cross_entropy_output_node_across_a_random_sweep():

    rng = random.Random(70)
    dimension = 4
    size = 5

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        cross_entropy_layer = CrossEntropyOutputLayer(size, state_layer)

        for node in cross_entropy_layer.nodes:
            node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
            node.bias = rng.uniform(-3.0, 3.0)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        cross_entropy_layer.forward()

        targets = [rng.choice([0.0, 1.0]) for _ in range(size)]
        expected = []
        for node, target in zip(cross_entropy_layer.nodes, targets):
            node.compute_output_delta(target)
            expected.append(node.delta)

        rust_layer = _snapshot_to_cross_entropy_rust_array_layer(cross_entropy_layer)
        rust_layer.forward(pa.Array(x))
        rust_layer.compute_output_delta(pa.Array(targets))

        assert np.allclose(rust_layer.delta.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_batch_matches_per_row_single_example_results_stacked():

    rng = random.Random(71)
    dimension = 4
    size = 3
    batch_size = 5

    rust_layer = CrossEntropyRustArrayLayer(size, dimension)
    rust_layer.W = pa.Array([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    rust_layer.b = pa.Array([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = [[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)]
    reference_rows = [[rng.choice([0.0, 1.0]) for _ in range(size)] for _ in range(batch_size)]

    expected_rows = []
    for row in range(batch_size):
        rust_layer.forward(pa.Array(X[row]))
        rust_layer.compute_output_delta(pa.Array(reference_rows[row]))
        expected_rows.append(rust_layer.delta.tolist())

    rust_layer.forward_batch(pa.Array(X))
    rust_layer.compute_output_delta_batch(pa.Array(reference_rows))
    assert np.allclose(rust_layer.delta_batch.tolist(), expected_rows, rtol=1e-9, atol=1e-12)


def test_forward_is_inherited_unchanged_from_rust_array_layer():

    assert CrossEntropyRustArrayLayer.__mro__[1] is RustArrayLayer
    assert CrossEntropyRustArrayLayer.forward is RustArrayLayer.forward
    assert CrossEntropyRustArrayLayer.forward_batch is RustArrayLayer.forward_batch


def test_compute_hidden_delta_is_inherited_unchanged_from_rust_array_layer():

    assert CrossEntropyRustArrayLayer.compute_hidden_delta is RustArrayLayer.compute_hidden_delta
    assert CrossEntropyRustArrayLayer.compute_hidden_delta_batch is RustArrayLayer.compute_hidden_delta_batch


def test_apply_accumulated_gradient_is_inherited_unchanged_from_rust_array_layer():

    rust_layer = CrossEntropyRustArrayLayer(2, 2)
    rust_layer.W = pa.Array([[1.0, 2.0], [3.0, 4.0]])
    rust_layer.b = pa.Array([5.0, 6.0])
    rust_layer.delta = pa.Array([1.0, 1.0])
    rust_layer.accumulate_gradient(pa.Array([1.0, 1.0]))
    rust_layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(rust_layer.W.tolist(), np.array([[0.9, 1.9], [2.9, 3.9]]))
    assert np.allclose(rust_layer.b.tolist(), np.array([4.9, 5.9]))
