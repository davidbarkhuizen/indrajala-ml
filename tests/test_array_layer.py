import random

import numpy as np
import pytest

from indrajala_ml.model.array_layer import ArrayLayer, sigmoid
from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.backprop_node import sigmoid as node_sigmoid
from indrajala_ml.model.state_layer import StateLayer
from tests.helpers import set_random_node_weights


def test_sigmoid_matches_node_sigmoid_across_a_random_sweep_including_the_overflow_boundary():

    # backprop_node.sigmoid's own docstring pins math.exp(710) as raising OverflowError and
    # math.exp(700) as not - both sides of that boundary, plus the ordinary range, get checked
    # here so numpy's inf-based overflow path is confirmed to land on the same limiting value,
    # not assumed from the formulas looking equivalent (see array_layer.sigmoid's docstring).
    rng = random.Random(0)
    z_values = [rng.uniform(-50.0, 50.0) for _ in range(200)]
    z_values += [-700.0, -709.0, -710.0, -1000.0, -1e10, 700.0, 709.0, 710.0, 1000.0, 0.0]

    for z in z_values:
        expected = node_sigmoid(z)
        actual = float(sigmoid(np.array([z]))[0])
        assert actual == pytest.approx(expected, abs=1e-12)


def _snapshot_to_array_layer(backprop_layer: BackpropLayer) -> ArrayLayer:
    array_layer = ArrayLayer(backprop_layer.size, len(backprop_layer.input_layer.nodes))
    snapshot = backprop_layer.snapshot_state()
    array_layer.W = np.array([weights for weights, _bias in snapshot])
    array_layer.b = np.array([bias for _weights, bias in snapshot])
    return array_layer


def test_forward_matches_backprop_layer_across_a_random_sweep():

    rng = random.Random(1)
    dimension = 5
    size = 4

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        backprop_layer = BackpropLayer(size, state_layer)

        set_random_node_weights(rng, backprop_layer, dimension, 3.0)

        array_layer = _snapshot_to_array_layer(backprop_layer)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        backprop_layer.forward()
        expected = [node.value() for node in backprop_layer.nodes]

        actual = array_layer.forward(np.array(x))

        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_forward_batch_matches_forward_run_once_per_row_and_stacked():

    rng = random.Random(2)
    dimension = 6
    size = 3
    batch_size = 8

    array_layer = ArrayLayer(size, dimension)
    array_layer.W = np.array([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    array_layer.b = np.array([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = np.array([[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)])

    expected = np.stack([array_layer.forward(x) for x in X])

    actual = array_layer.forward_batch(X)

    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_matches_node_compute_output_delta_across_a_random_sweep():

    rng = random.Random(3)

    for _ in range(200):
        a = rng.uniform(0.0, 1.0)
        reference = rng.uniform(0.0, 1.0)

        node = BackpropNode(input_nodes=[])
        node._activation = a
        node.compute_output_delta(reference)

        array_layer = ArrayLayer(1, 0)
        array_layer.a = np.array([a])
        array_layer.compute_output_delta(np.array([reference]))

        assert array_layer.delta[0] == pytest.approx(node.delta, abs=1e-12)


def test_compute_hidden_delta_matches_node_compute_hidden_delta_across_a_random_sweep():

    rng = random.Random(4)
    hidden_size = 5
    next_size = 4

    for _ in range(100):
        state_layer = StateLayer(hidden_size, [(-10.0, 10.0)] * hidden_size)
        hidden_layer = BackpropLayer(hidden_size, state_layer)
        next_layer = BackpropLayer(next_size, hidden_layer)

        for node in next_layer.nodes:
            node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(hidden_size)])
            node.bias = rng.uniform(-3.0, 3.0)
            node.delta = rng.uniform(-5.0, 5.0)

        activations = [rng.uniform(0.0, 1.0) for _ in range(hidden_size)]
        for node, a in zip(hidden_layer.nodes, activations):
            node._activation = a

        expected = []
        for i, node in enumerate(hidden_layer.nodes):
            node.compute_hidden_delta(next_layer.nodes, i)
            expected.append(node.delta)

        array_hidden = ArrayLayer(hidden_size, hidden_size)
        array_hidden.a = np.array(activations)
        array_next = _snapshot_to_array_layer(next_layer)
        array_next.delta = np.array([node.delta for node in next_layer.nodes])

        array_hidden.compute_hidden_delta(array_next)

        assert np.allclose(array_hidden.delta, expected, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_batch_matches_per_row_single_example_results_stacked():

    rng = random.Random(5)
    size = 4
    batch_size = 6

    array_layer = ArrayLayer(size, 0)
    A = np.array([[rng.uniform(0.0, 1.0) for _ in range(size)] for _ in range(batch_size)])
    reference_batch = np.array([[rng.uniform(0.0, 1.0) for _ in range(size)] for _ in range(batch_size)])

    expected_rows = []
    for a_row, reference_row in zip(A, reference_batch):
        array_layer.a = a_row
        array_layer.compute_output_delta(reference_row)
        expected_rows.append(array_layer.delta)
    expected = np.stack(expected_rows)

    array_layer.A = A
    array_layer.compute_output_delta_batch(reference_batch)

    assert np.allclose(array_layer.delta_batch, expected, rtol=1e-9, atol=1e-12)


def test_compute_hidden_delta_batch_matches_per_row_single_example_results_stacked():

    rng = random.Random(6)
    hidden_size = 5
    next_size = 3
    batch_size = 7

    next_layer = ArrayLayer(next_size, hidden_size)
    next_layer.W = np.array([[rng.uniform(-3.0, 3.0) for _ in range(hidden_size)] for _ in range(next_size)])
    next_layer.delta_batch = np.array(
        [[rng.uniform(-5.0, 5.0) for _ in range(next_size)] for _ in range(batch_size)]
    )

    hidden_layer = ArrayLayer(hidden_size, 0)
    A = np.array([[rng.uniform(0.0, 1.0) for _ in range(hidden_size)] for _ in range(batch_size)])

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


def test_accumulate_then_apply_at_batch_size_one_matches_backprop_node_across_a_random_sweep():

    rng = random.Random(7)
    dimension = 5
    size = 4

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        backprop_layer = BackpropLayer(size, state_layer)

        for node in backprop_layer.nodes:
            node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
            node.bias = rng.uniform(-3.0, 3.0)
            node.delta = rng.uniform(-5.0, 5.0)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        learning_rate = rng.uniform(0.001, 1.0)

        array_layer = _snapshot_to_array_layer(backprop_layer)
        array_layer.delta = np.array([node.delta for node in backprop_layer.nodes])

        for node in backprop_layer.nodes:
            node.accumulate_gradient()
            node.apply_accumulated_gradient(learning_rate, batch_size=1)
        array_layer.accumulate_gradient(np.array(x))
        array_layer.apply_accumulated_gradient(learning_rate, batch_size=1)

        expected_W = np.array([node.input_node_weights for node in backprop_layer.nodes])
        expected_b = np.array([node.bias for node in backprop_layer.nodes])

        assert np.allclose(array_layer.W, expected_W, rtol=1e-9, atol=1e-12)
        assert np.allclose(array_layer.b, expected_b, rtol=1e-9, atol=1e-12)


def test_accumulate_across_a_batch_then_apply_matches_backprop_node_across_a_random_sweep():

    # multiple examples accumulated (different input, different delta each time) before any
    # weight is written, then one apply at batch_size>1 - mirrors mini-batch gradient descent's
    # own accumulate/apply split (see tests/test_gradient_accumulation.py's batched cases)
    rng = random.Random(8)
    dimension = 4
    size = 3
    batch_size = 6

    state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
    backprop_layer = BackpropLayer(size, state_layer)
    for node in backprop_layer.nodes:
        node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
        node.bias = rng.uniform(-3.0, 3.0)

    array_layer = _snapshot_to_array_layer(backprop_layer)
    learning_rate = rng.uniform(0.001, 1.0)

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

    array_layer = ArrayLayer(2, 3)
    array_layer.delta = np.array([0.2, -0.1])
    array_layer.accumulate_gradient(np.array([1.0, 1.0, 1.0]))
    array_layer.apply_accumulated_gradient(0.1, batch_size=1)

    weights_after_first_apply = array_layer.W.copy()
    bias_after_first_apply = array_layer.b.copy()

    # a second apply with nothing accumulated in between must be a no-op
    array_layer.apply_accumulated_gradient(0.1, batch_size=1)

    assert np.array_equal(array_layer.W, weights_after_first_apply)
    assert np.array_equal(array_layer.b, bias_after_first_apply)


def test_accumulate_gradient_batch_matches_looping_accumulate_gradient_over_every_row():

    rng = random.Random(9)
    size = 3
    input_size = 4
    batch_size = 5

    looped = ArrayLayer(size, input_size)
    one_shot = ArrayLayer(size, input_size)

    delta_batch = np.array([[rng.uniform(-5.0, 5.0) for _ in range(size)] for _ in range(batch_size)])
    X = np.array([[rng.uniform(-10.0, 10.0) for _ in range(input_size)] for _ in range(batch_size)])

    for row in range(batch_size):
        looped.delta = delta_batch[row]
        looped.accumulate_gradient(X[row])

    one_shot.delta_batch = delta_batch
    one_shot.accumulate_gradient_batch(X)

    assert np.allclose(one_shot._grad_W, looped._grad_W, rtol=1e-9, atol=1e-12)
    assert np.allclose(one_shot._grad_b, looped._grad_b, rtol=1e-9, atol=1e-12)


def test_downstream_matches_the_hand_written_transpose_matmul_single_and_batch():

    # downstream()/downstream_batch() are the gradient an ArrayLayer sends back to its input -
    # pinned here against the expressions every compute_hidden_delta* site used to inline
    # (next_layer.W.T @ next_layer.delta and next_layer.delta_batch @ next_layer.W)
    rng = np.random.default_rng(0)
    layer = ArrayLayer(4, 6)
    layer.W = rng.uniform(-1.0, 1.0, size=(4, 6))
    layer.delta = rng.uniform(-1.0, 1.0, size=4)
    layer.delta_batch = rng.uniform(-1.0, 1.0, size=(3, 4))

    np.testing.assert_array_equal(layer.downstream(), layer.W.T @ layer.delta)
    np.testing.assert_array_equal(layer.downstream_batch(), layer.delta_batch @ layer.W)
    assert layer.downstream().shape == (6,)
    assert layer.downstream_batch().shape == (3, 6)
