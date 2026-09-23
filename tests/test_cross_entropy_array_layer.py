import random

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.binary_cross_entropy_backprop_classifier_network import CrossEntropyOutputLayer
from indrajala_ml.model.cross_entropy_array_layer import CrossEntropyArrayLayer
from indrajala_ml.model.state_layer import StateLayer


def _snapshot_to_cross_entropy_array_layer(cross_entropy_layer: CrossEntropyOutputLayer) -> CrossEntropyArrayLayer:
    array_layer = CrossEntropyArrayLayer(cross_entropy_layer.size, len(cross_entropy_layer.input_layer.nodes))
    snapshot = cross_entropy_layer.snapshot_state()
    array_layer.W = np.array([weights for weights, _bias in snapshot])
    array_layer.b = np.array([bias for _weights, bias in snapshot])
    return array_layer


def test_compute_output_delta_matches_cross_entropy_output_node_across_a_random_sweep():

    rng = random.Random(60)
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

        array_layer = _snapshot_to_cross_entropy_array_layer(cross_entropy_layer)
        array_layer.forward(np.array(x))
        array_layer.compute_output_delta(np.array(targets))

        assert np.allclose(array_layer.delta, expected, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_batch_matches_per_row_single_example_results_stacked():

    rng = random.Random(61)
    dimension = 4
    size = 3
    batch_size = 5

    array_layer = CrossEntropyArrayLayer(size, dimension)
    array_layer.W = np.array([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    array_layer.b = np.array([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = np.array([[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)])
    reference_batch = np.array([[rng.choice([0.0, 1.0]) for _ in range(size)] for _ in range(batch_size)])

    expected_rows = []
    for row in range(batch_size):
        array_layer.forward(X[row])
        array_layer.compute_output_delta(reference_batch[row])
        expected_rows.append(array_layer.delta)
    expected = np.stack(expected_rows)

    array_layer.forward_batch(X)
    array_layer.compute_output_delta_batch(reference_batch)
    assert np.allclose(array_layer.delta_batch, expected, rtol=1e-9, atol=1e-12)


def test_forward_is_inherited_unchanged_from_array_layer():

    assert CrossEntropyArrayLayer.__mro__[1] is ArrayLayer
    assert CrossEntropyArrayLayer.forward is ArrayLayer.forward
    assert CrossEntropyArrayLayer.forward_batch is ArrayLayer.forward_batch


def test_compute_hidden_delta_is_inherited_unchanged_from_array_layer():

    assert CrossEntropyArrayLayer.compute_hidden_delta is ArrayLayer.compute_hidden_delta
    assert CrossEntropyArrayLayer.compute_hidden_delta_batch is ArrayLayer.compute_hidden_delta_batch


def test_apply_accumulated_gradient_is_inherited_unchanged_from_array_layer():

    array_layer = CrossEntropyArrayLayer(2, 2)
    array_layer.W = np.array([[1.0, 2.0], [3.0, 4.0]])
    array_layer.b = np.array([5.0, 6.0])
    array_layer.delta = np.array([1.0, 1.0])
    array_layer.accumulate_gradient(np.array([1.0, 1.0]))
    array_layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(array_layer.W, np.array([[0.9, 1.9], [2.9, 3.9]]))
    assert np.allclose(array_layer.b, np.array([4.9, 5.9]))
