"""
The crate's activation family (layer_activation_forward*, layer_activation_hidden_delta*; the FFN
activations workplan, D9), one fused Rust call per ReLUArrayLayer or GELUArrayLayer method, checked
against those numpy layers, the production reference these functions replace, the same treatment
test_fused_layer_ops.py gives every plain fused op. numpy's exp is the crate's (in activations.py
and erf.py), so the activations agree but for the products' rounding.
"""

import random

import numpy as np
import pytest
from indrajala_math_rust import (
    Array,
    layer_activation_forward,
    layer_activation_forward_batch,
    layer_activation_hidden_delta,
    layer_activation_hidden_delta_batch,
)

from indrajala_ml.model.layers.numpy import activations, erf
from indrajala_ml.model.layers.numpy.array_layer import ArrayLayer
from indrajala_ml.model.layers.numpy.gelu_array_layer import GELUArrayLayer
from indrajala_ml.model.layers.numpy.relu_array_layer import ReLUArrayLayer
from indrajala_ml.model.layers.python.activations import HiddenActivation
from tests.helpers import approx, exp_by_crate, random_matrix, random_vector, rust_to_numpy

SEEDS = range(10)
ACTIVATIONS: list[HiddenActivation] = ["relu", "gelu", "gelu_tanh"]
INPUT_SIZE = 8
HIDDEN_SIZE = 5
NEXT_SIZE = 4
BATCH_SIZE = 6


@pytest.fixture(autouse=True)
def crate_exp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(activations, "exp", exp_by_crate)
    monkeypatch.setattr(erf, "exp", exp_by_crate)


def _layer(activation: HiddenActivation, size: int, input_size: int) -> ArrayLayer:
    if activation == "relu":
        return ReLUArrayLayer(size, input_size)
    return GELUArrayLayer(size, input_size, activation)


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_the_forward_matches_the_numpy_layers(activation: HiddenActivation, seed: int):
    rng = random.Random(seed)
    w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = random_vector(rng, HIDDEN_SIZE)
    x_data = random_vector(rng, INPUT_SIZE)
    X_data = random_matrix(rng, BATCH_SIZE, INPUT_SIZE)

    layer = _layer(activation, HIDDEN_SIZE, INPUT_SIZE)
    layer.W, layer.b = np.array(w_data), np.array(b_data)

    a, z = layer_activation_forward(Array(w_data), Array(x_data), Array(b_data), activation)
    assert rust_to_numpy(a) == approx(layer.forward(np.array(x_data)))
    A, Z = layer_activation_forward_batch(Array(w_data), Array(X_data), Array(b_data), activation)
    assert rust_to_numpy(A) == approx(layer.forward_batch(np.array(X_data)))
    if activation == "relu":
        assert z is None and Z is None
    else:
        assert z is not None and rust_to_numpy(z) == approx(layer.z)
        assert Z is not None and rust_to_numpy(Z) == approx(layer.Z)


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_the_hidden_deltas_match_the_numpy_layers(activation: HiddenActivation, seed: int):
    rng = random.Random(seed)
    next_w_data = random_matrix(rng, NEXT_SIZE, HIDDEN_SIZE)
    next_delta_data = random_vector(rng, NEXT_SIZE)
    next_delta_batch_data = random_matrix(rng, BATCH_SIZE, NEXT_SIZE)
    # ReLU's cached operand is a, the GELUs' z: either way values about zero
    cached = [rng.uniform(-3.0, 3.0) for _ in range(HIDDEN_SIZE)]
    cached_batch = [[rng.uniform(-3.0, 3.0) for _ in range(HIDDEN_SIZE)] for _ in range(BATCH_SIZE)]

    this_layer = _layer(activation, HIDDEN_SIZE, INPUT_SIZE)
    if activation == "relu":
        this_layer.a, this_layer.A = np.array(cached), np.array(cached_batch)
    else:
        this_layer.z, this_layer.Z = np.array(cached), np.array(cached_batch)
    next_layer = ReLUArrayLayer(NEXT_SIZE, HIDDEN_SIZE)
    next_layer.W = np.array(next_w_data)
    next_layer.delta, next_layer.delta_batch = np.array(next_delta_data), np.array(next_delta_batch_data)
    this_layer.compute_hidden_delta(next_layer)
    this_layer.compute_hidden_delta_batch(next_layer)

    actual = layer_activation_hidden_delta(Array(next_w_data), Array(next_delta_data), Array(cached), activation)
    assert rust_to_numpy(actual) == approx(this_layer.delta)
    actual_batch = layer_activation_hidden_delta_batch(
        Array(next_w_data), Array(next_delta_batch_data), Array(cached_batch), activation
    )
    assert rust_to_numpy(actual_batch) == approx(this_layer.delta_batch)


def test_the_relu_hidden_delta_is_exactly_zero_at_the_activation_equals_zero_boundary():
    result = layer_activation_hidden_delta(Array([[3.0]]), Array([7.0]), Array([0.0]), "relu")
    assert result[0] == 0.0


def test_the_hidden_delta_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        layer_activation_hidden_delta(Array.zeros((2, 3)), Array.zeros(2), Array.zeros(5), "relu")
