"""
layer_relu_forward/layer_relu_forward_batch/layer_relu_hidden_delta/layer_relu_hidden_delta_batch,
one fused Rust call per ReLUArrayLayer method, checked against
indrajala_ml.model.relu_array_layer.ReLUArrayLayer
- the actual production reference these functions replace - the same treatment
test_fused_layer_ops.py gives every plain (non-activation-changing) fused op.
"""

import random

import numpy as np
import pytest
from indrajala_math_rust import (
    Array,
    layer_relu_forward,
    layer_relu_forward_batch,
    layer_relu_hidden_delta,
    layer_relu_hidden_delta_batch,
)

from indrajala_ml.model.relu_array_layer import ReLUArrayLayer
from tests.helpers import approx, random_matrix, random_vector, rust_to_numpy

SEEDS = range(30)
INPUT_SIZE = 8
HIDDEN_SIZE = 5
NEXT_SIZE = 4
BATCH_SIZE = 6


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_relu_forward_matches_relu_array_layer_forward(seed: int):
    rng = random.Random(seed)
    w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = random_vector(rng, HIDDEN_SIZE)
    x_data = random_vector(rng, INPUT_SIZE)

    layer = ReLUArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    expected = layer.forward(np.array(x_data))

    actual = layer_relu_forward(Array(w_data), Array(x_data), Array(b_data))
    assert rust_to_numpy(actual) == approx(expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_relu_forward_batch_matches_relu_array_layer_forward_batch(seed: int):
    rng = random.Random(seed)
    w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = random_vector(rng, HIDDEN_SIZE)
    x_data = random_matrix(rng, BATCH_SIZE, INPUT_SIZE)

    layer = ReLUArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    expected = layer.forward_batch(np.array(x_data))

    actual = layer_relu_forward_batch(Array(w_data), Array(x_data), Array(b_data))
    assert rust_to_numpy(actual) == approx(expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_relu_hidden_delta_matches_relu_array_layer_compute_hidden_delta(seed: int):
    rng = random.Random(seed)
    next_w_data = random_matrix(rng, NEXT_SIZE, HIDDEN_SIZE)
    next_delta_data = random_vector(rng, NEXT_SIZE)
    a_data = [rng.uniform(-3.0, 3.0) for _ in range(HIDDEN_SIZE)]

    this_layer = ReLUArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    this_layer.a = np.array(a_data)
    next_layer = ReLUArrayLayer(NEXT_SIZE, HIDDEN_SIZE)
    next_layer.W = np.array(next_w_data)
    next_layer.delta = np.array(next_delta_data)
    this_layer.compute_hidden_delta(next_layer)

    actual = layer_relu_hidden_delta(Array(next_w_data), Array(next_delta_data), Array(a_data))
    assert rust_to_numpy(actual) == approx(this_layer.delta)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_relu_hidden_delta_batch_matches_relu_array_layer_compute_hidden_delta_batch(seed: int):
    rng = random.Random(seed)
    next_w_data = random_matrix(rng, NEXT_SIZE, HIDDEN_SIZE)
    next_delta_batch_data = random_matrix(rng, BATCH_SIZE, NEXT_SIZE)
    a_batch_data = [[rng.uniform(-3.0, 3.0) for _ in range(HIDDEN_SIZE)] for _ in range(BATCH_SIZE)]

    this_layer = ReLUArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    this_layer.A = np.array(a_batch_data)
    next_layer = ReLUArrayLayer(NEXT_SIZE, HIDDEN_SIZE)
    next_layer.W = np.array(next_w_data)
    next_layer.delta_batch = np.array(next_delta_batch_data)
    this_layer.compute_hidden_delta_batch(next_layer)

    actual = layer_relu_hidden_delta_batch(Array(next_w_data), Array(next_delta_batch_data), Array(a_batch_data))
    assert rust_to_numpy(actual) == approx(this_layer.delta_batch)


def test_layer_relu_hidden_delta_is_exactly_zero_at_the_activation_equals_zero_boundary():
    result = layer_relu_hidden_delta(Array([[3.0]]), Array([7.0]), Array([0.0]))
    assert result[0] == 0.0


def test_layer_relu_hidden_delta_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        layer_relu_hidden_delta(Array.zeros((2, 3)), Array.zeros(2), Array.zeros(5))
