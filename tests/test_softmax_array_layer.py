import random

import numpy as np
import pytest

from indrajala_ml.model.softmax_array_layer import SoftmaxArrayLayer
from indrajala_ml.model.softmax_output_layer import SoftmaxOutputLayer
from indrajala_ml.model.state_layer import StateLayer
from tests.helpers import set_random_node_weights


def _snapshot_to_softmax_array_layer(softmax_layer: SoftmaxOutputLayer) -> SoftmaxArrayLayer:
    array_layer = SoftmaxArrayLayer(softmax_layer.size, len(softmax_layer.input_layer.nodes))
    snapshot = softmax_layer.snapshot_state()
    array_layer.W = np.array([weights for weights, _bias in snapshot])
    array_layer.b = np.array([bias for _weights, bias in snapshot])
    return array_layer


def test_forward_matches_softmax_output_layer_across_a_random_sweep():

    rng = random.Random(50)
    dimension = 4
    size = 5

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        softmax_layer = SoftmaxOutputLayer(size, state_layer)

        set_random_node_weights(rng, softmax_layer, dimension, 3.0)

        array_layer = _snapshot_to_softmax_array_layer(softmax_layer)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        softmax_layer.forward()
        expected = [node.value() for node in softmax_layer.nodes]

        actual = array_layer.forward(np.array(x))
        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
        assert actual.sum() == pytest.approx(1.0)


def test_forward_matches_softmax_output_layer_for_large_magnitude_z_without_overflow():

    # the numerically-adversarial case: confirms the array layer's max-shift trick matches the
    # per-node reference's own overflow-safe behavior, not just the well-conditioned case
    dimension = 1
    state_layer = StateLayer(dimension, [(-1.0, 1.0)])
    softmax_layer = SoftmaxOutputLayer(3, state_layer)
    for node, (weight, bias) in zip(softmax_layer.nodes, [(1.0, 10_000.0), (1.0, 0.0), (1.0, -10_000.0)]):
        node.update_input_weights([weight])
        node.bias = bias
    state_layer.update_state((0.0,))

    array_layer = _snapshot_to_softmax_array_layer(softmax_layer)

    softmax_layer.forward()
    expected = [node.value() for node in softmax_layer.nodes]

    actual = array_layer.forward(np.array([0.0]))
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
    assert actual[0] == pytest.approx(1.0)
    assert actual[1] == pytest.approx(0.0)
    assert actual[2] == pytest.approx(0.0)
    assert not np.isnan(actual).any()


def test_forward_batch_matches_per_row_single_example_results_stacked():

    rng = random.Random(51)
    dimension = 4
    size = 3
    batch_size = 6

    array_layer = SoftmaxArrayLayer(size, dimension)
    array_layer.W = np.array([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    array_layer.b = np.array([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = np.array([[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)])

    expected_rows = [array_layer.forward(x) for x in X]
    expected = np.stack(expected_rows)

    actual = array_layer.forward_batch(X)
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
    assert np.allclose(actual.sum(axis=1), 1.0)


def test_compute_output_delta_matches_softmax_output_node_across_a_random_sweep():

    rng = random.Random(52)
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

        array_layer = _snapshot_to_softmax_array_layer(softmax_layer)
        array_layer.forward(np.array(x))
        reference = np.zeros(size)
        reference[category] = 1.0
        array_layer.compute_output_delta(reference)

        assert np.allclose(array_layer.delta, expected, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_batch_matches_per_row_single_example_results_stacked():

    rng = random.Random(53)
    dimension = 4
    size = 3
    batch_size = 5

    array_layer = SoftmaxArrayLayer(size, dimension)
    array_layer.W = np.array([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    array_layer.b = np.array([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = np.array([[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)])
    categories = [rng.randrange(size) for _ in range(batch_size)]
    reference_batch = np.zeros((batch_size, size))
    for row, category in enumerate(categories):
        reference_batch[row, category] = 1.0

    expected_rows = []
    for row in range(batch_size):
        array_layer.forward(X[row])
        array_layer.compute_output_delta(reference_batch[row])
        expected_rows.append(array_layer.delta)
    expected = np.stack(expected_rows)

    array_layer.forward_batch(X)
    array_layer.compute_output_delta_batch(reference_batch)
    assert np.allclose(array_layer.delta_batch, expected, rtol=1e-9, atol=1e-12)


def test_compute_hidden_delta_is_inherited_unchanged_from_array_layer():

    # softmax's cross-node coupling only affects the forward pass - whatever layer feeds this
    # one still uses the plain sigmoid compute_hidden_delta formula
    from indrajala_ml.model.array_layer import ArrayLayer

    hidden = SoftmaxArrayLayer.__mro__[1]
    assert hidden is ArrayLayer
    assert SoftmaxArrayLayer.compute_hidden_delta is ArrayLayer.compute_hidden_delta
    assert SoftmaxArrayLayer.compute_hidden_delta_batch is ArrayLayer.compute_hidden_delta_batch


def test_apply_accumulated_gradient_is_inherited_unchanged_from_array_layer():

    array_layer = SoftmaxArrayLayer(2, 2)
    array_layer.W = np.array([[1.0, 2.0], [3.0, 4.0]])
    array_layer.b = np.array([5.0, 6.0])
    array_layer.delta = np.array([1.0, 1.0])
    array_layer.accumulate_gradient(np.array([1.0, 1.0]))
    array_layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(array_layer.W, np.array([[0.9, 1.9], [2.9, 3.9]]))
    assert np.allclose(array_layer.b, np.array([4.9, 5.9]))


def test_construction_rejects_a_size_smaller_than_two():

    with pytest.raises(AssertionError):
        SoftmaxArrayLayer(1, 3)
