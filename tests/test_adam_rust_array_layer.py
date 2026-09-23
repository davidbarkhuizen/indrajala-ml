import random

import numpy as np
import pytest

import indrajala_math_rust as pa
from indrajala_ml.model.adam_layer import make_adam_layer_cls
from indrajala_ml.model.adam_rust_array_layer import AdamRustArrayLayer
from indrajala_ml.model.state_layer import StateLayer

BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8


def _snapshot_to_adam_rust_array_layer(backprop_layer) -> AdamRustArrayLayer:
    rust_layer = AdamRustArrayLayer(backprop_layer.size, len(backprop_layer.input_layer.nodes), BETA1, BETA2, EPSILON)
    snapshot = backprop_layer.snapshot_state()
    rust_layer.W = pa.Array([weights for weights, _bias in snapshot])
    rust_layer.b = pa.Array([bias for _weights, bias in snapshot])
    return rust_layer


def test_accumulate_then_apply_at_batch_size_one_matches_adam_backprop_node_at_every_step():

    # mirrors test_adam_array_layer.py's own
    # test_accumulate_then_apply_at_batch_size_one_matches_adam_backprop_node_at_every_step,
    # against the Rust-matmul-backed AdamRustArrayLayer instead of the numpy AdamArrayLayer
    rng = random.Random(11)
    dimension = 5
    size = 4
    adam_layer_cls = make_adam_layer_cls(BETA1, BETA2, EPSILON)

    state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
    backprop_layer = adam_layer_cls(size=size, input_layer=state_layer)
    for node in backprop_layer.nodes:
        node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
        node.bias = rng.uniform(-3.0, 3.0)

    rust_layer = _snapshot_to_adam_rust_array_layer(backprop_layer)
    learning_rate = rng.uniform(0.001, 1.0)

    for _ in range(10):
        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        deltas = [rng.uniform(-5.0, 5.0) for _ in range(size)]

        for node, delta in zip(backprop_layer.nodes, deltas):
            node.delta = delta
            node.accumulate_gradient()
            node.apply_accumulated_gradient(learning_rate, batch_size=1)

        rust_layer.delta = pa.Array(deltas)
        rust_layer.accumulate_gradient(pa.Array(x))
        rust_layer.apply_accumulated_gradient(learning_rate, batch_size=1)

        expected_W = np.array([node.input_node_weights for node in backprop_layer.nodes])
        expected_b = np.array([node.bias for node in backprop_layer.nodes])
        assert np.allclose(rust_layer.W.tolist(), expected_W, rtol=1e-9, atol=1e-12)
        assert np.allclose(rust_layer.b.tolist(), expected_b, rtol=1e-9, atol=1e-12)


def test_accumulate_across_a_batch_then_apply_matches_adam_backprop_node_at_every_batch():

    # mirrors test_adam_array_layer.py's own
    # test_accumulate_across_a_batch_then_apply_matches_adam_backprop_node_at_every_batch
    rng = random.Random(12)
    dimension = 4
    size = 3
    batch_size = 6
    adam_layer_cls = make_adam_layer_cls(BETA1, BETA2, EPSILON)

    state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
    backprop_layer = adam_layer_cls(size=size, input_layer=state_layer)
    for node in backprop_layer.nodes:
        node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
        node.bias = rng.uniform(-3.0, 3.0)

    rust_layer = _snapshot_to_adam_rust_array_layer(backprop_layer)
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

            rust_layer.delta = pa.Array(deltas)
            rust_layer.accumulate_gradient(pa.Array(x))

        for node in backprop_layer.nodes:
            node.apply_accumulated_gradient(learning_rate, batch_size)
        rust_layer.apply_accumulated_gradient(learning_rate, batch_size)

        expected_W = np.array([node.input_node_weights for node in backprop_layer.nodes])
        expected_b = np.array([node.bias for node in backprop_layer.nodes])
        assert np.allclose(rust_layer.W.tolist(), expected_W, rtol=1e-9, atol=1e-12)
        assert np.allclose(rust_layer.b.tolist(), expected_b, rtol=1e-9, atol=1e-12)


def test_step_count_increments_once_per_apply_call():

    rust_layer = AdamRustArrayLayer(3, 2, BETA1, BETA2, EPSILON)
    assert rust_layer._t == 0

    rust_layer.delta = pa.Array([0.1, 0.2, 0.3])
    rust_layer.accumulate_gradient(pa.Array([1.0, 2.0]))
    rust_layer.apply_accumulated_gradient(0.1, batch_size=1)
    assert rust_layer._t == 1

    rust_layer.delta = pa.Array([0.1, 0.2, 0.3])
    rust_layer.accumulate_gradient(pa.Array([1.0, 2.0]))
    rust_layer.apply_accumulated_gradient(0.1, batch_size=1)
    assert rust_layer._t == 2


def test_apply_accumulated_gradient_resets_the_accumulator():

    rust_layer = AdamRustArrayLayer(3, 2, BETA1, BETA2, EPSILON)
    rust_layer.delta = pa.Array([0.1, 0.2, 0.3])
    rust_layer.accumulate_gradient(pa.Array([1.0, 2.0]))
    rust_layer.apply_accumulated_gradient(0.1, batch_size=1)

    assert np.allclose(rust_layer._grad_W.tolist(), np.zeros((3, 2)))
    assert np.allclose(rust_layer._grad_b.tolist(), np.zeros(3))
