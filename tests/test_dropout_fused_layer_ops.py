"""
layer_dropout_forward/layer_dropout_forward_batch/layer_dropout_hidden_delta/
layer_dropout_hidden_delta_batch, checked against
indrajala_ml.model.dropout_array_layer.DropoutArrayLayer - the actual production reference these
functions replace - the same treatment test_relu_fused_layer_ops.py gives ReLUArrayLayer's own
fused ops.

training=True is checked against the reference too: the crate's RNG is numpy's np.random in a
separate state, so after np.random.seed(s) and pa.seed(s) both draw the same masks, bit for bit.
The training-mode tests seed both, then compare forward, forward_batch and a full learn_batch step
of DropoutRustArrayLayer with DropoutArrayLayer, masks exactly and values to the training=False
tests' bar. layer_dropout_hidden_delta is also checked with a forced mask/base_activation pair,
against the hand-derived chain rule.
"""

import random
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest
from indrajala_math_rust import (
    Array,
    layer_dropout_forward,
    layer_dropout_forward_batch,
    layer_dropout_hidden_delta,
    layer_dropout_hidden_delta_batch,
)

from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.dropout_array_layer import DropoutArrayLayer
from indrajala_ml.model.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from tests.helpers import approx, random_matrix, random_vector, rust_to_numpy

SEEDS = range(30)
INPUT_SIZE = 8
HIDDEN_SIZE = 5
NEXT_SIZE = 4
BATCH_SIZE = 6
DROP_PROBABILITY = 0.4
KEEP_PROBABILITY = 1.0 - DROP_PROBABILITY


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_dropout_forward_at_eval_mode_matches_dropout_array_layer_forward(seed: int):
    rng = random.Random(seed)
    w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = random_vector(rng, HIDDEN_SIZE)
    x_data = random_vector(rng, INPUT_SIZE)

    layer = DropoutArrayLayer(HIDDEN_SIZE, INPUT_SIZE, DROP_PROBABILITY)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    expected = layer.forward(np.array(x_data))  # training=False by default

    a, mask, base_activation = layer_dropout_forward(
        Array(w_data), Array(x_data), Array(b_data), DROP_PROBABILITY, False
    )
    assert rust_to_numpy(a) == approx(expected)
    assert rust_to_numpy(mask).tolist() == [1.0] * HIDDEN_SIZE
    assert rust_to_numpy(base_activation) == approx(expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_dropout_forward_batch_at_eval_mode_matches_dropout_array_layer_forward_batch(seed: int):
    rng = random.Random(seed)
    w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = random_vector(rng, HIDDEN_SIZE)
    x_data = random_matrix(rng, BATCH_SIZE, INPUT_SIZE)

    layer = DropoutArrayLayer(HIDDEN_SIZE, INPUT_SIZE, DROP_PROBABILITY)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    expected = layer.forward_batch(np.array(x_data))

    a, mask, base_activation = layer_dropout_forward_batch(
        Array(w_data), Array(x_data), Array(b_data), DROP_PROBABILITY, False
    )
    assert rust_to_numpy(a) == approx(expected)
    assert rust_to_numpy(base_activation) == approx(expected)
    assert mask.shape == (BATCH_SIZE, HIDDEN_SIZE)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_dropout_hidden_delta_at_eval_mode_matches_dropout_array_layer_compute_hidden_delta(seed: int):
    rng = random.Random(seed)
    next_w_data = random_matrix(rng, NEXT_SIZE, HIDDEN_SIZE)
    next_delta_data = random_vector(rng, NEXT_SIZE)
    a_data = [rng.uniform(0.01, 0.99) for _ in range(HIDDEN_SIZE)]

    this_layer = DropoutArrayLayer(HIDDEN_SIZE, INPUT_SIZE, DROP_PROBABILITY)
    this_layer._base_activation = np.array(a_data)
    this_layer._mask = np.ones(HIDDEN_SIZE)
    this_layer._was_training = False
    next_layer = DropoutArrayLayer(NEXT_SIZE, HIDDEN_SIZE, DROP_PROBABILITY)
    next_layer.W = np.array(next_w_data)
    next_layer.delta = np.array(next_delta_data)
    this_layer.compute_hidden_delta(next_layer)

    actual = layer_dropout_hidden_delta(
        Array(next_w_data),
        Array(next_delta_data),
        Array(a_data),
        Array([1.0] * HIDDEN_SIZE),
        KEEP_PROBABILITY,
        False,
    )
    assert rust_to_numpy(actual) == approx(this_layer.delta)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_dropout_hidden_delta_batch_at_eval_mode_matches_dropout_array_layer_compute_hidden_delta_batch(
    seed: int,
):
    rng = random.Random(seed)
    next_w_data = random_matrix(rng, NEXT_SIZE, HIDDEN_SIZE)
    next_delta_batch_data = random_matrix(rng, BATCH_SIZE, NEXT_SIZE)
    a_batch_data = [[rng.uniform(0.01, 0.99) for _ in range(HIDDEN_SIZE)] for _ in range(BATCH_SIZE)]
    ones_batch = [[1.0] * HIDDEN_SIZE for _ in range(BATCH_SIZE)]

    this_layer = DropoutArrayLayer(HIDDEN_SIZE, INPUT_SIZE, DROP_PROBABILITY)
    this_layer._base_activation_batch = np.array(a_batch_data)
    this_layer._mask_batch = np.ones((BATCH_SIZE, HIDDEN_SIZE))
    this_layer._was_training = False
    next_layer = DropoutArrayLayer(NEXT_SIZE, HIDDEN_SIZE, DROP_PROBABILITY)
    next_layer.W = np.array(next_w_data)
    next_layer.delta_batch = np.array(next_delta_batch_data)
    this_layer.compute_hidden_delta_batch(next_layer)

    actual = layer_dropout_hidden_delta_batch(
        Array(next_w_data),
        Array(next_delta_batch_data),
        Array(a_batch_data),
        Array(ones_batch),
        KEEP_PROBABILITY,
        False,
    )
    assert rust_to_numpy(actual) == approx(this_layer.delta_batch)


def _training_layers(rng: random.Random) -> tuple[DropoutArrayLayer, DropoutRustArrayLayer]:
    # the same weights on both backends, in training mode
    w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = random_vector(rng, HIDDEN_SIZE)
    numpy_layer = DropoutArrayLayer(HIDDEN_SIZE, INPUT_SIZE, DROP_PROBABILITY)
    numpy_layer.W, numpy_layer.b = np.array(w_data), np.array(b_data)
    rust_layer = DropoutRustArrayLayer(HIDDEN_SIZE, INPUT_SIZE, DROP_PROBABILITY)
    rust_layer.W, rust_layer.b = Array(w_data), Array(b_data)
    numpy_layer.set_training_mode(True)
    rust_layer.set_training_mode(True)
    return numpy_layer, rust_layer


def _seed_both(seed: int) -> None:
    np.random.seed(seed)
    pa.seed(seed)


@pytest.mark.parametrize("seed", SEEDS)
def test_forward_in_training_mode_matches_dropout_array_layer_after_the_same_seed(seed: int):
    rng = random.Random(seed)
    numpy_layer, rust_layer = _training_layers(rng)
    _seed_both(seed)
    for _ in range(5):  # consecutive passes: the position carries across calls
        x_data = random_vector(rng, INPUT_SIZE)
        expected = numpy_layer.forward(np.array(x_data))
        actual = rust_layer.forward(Array(x_data))
        assert rust_to_numpy(rust_layer._mask).tolist() == numpy_layer._mask.tolist()
        assert rust_to_numpy(rust_layer._base_activation) == approx(numpy_layer._base_activation)
        assert rust_to_numpy(actual) == approx(expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_forward_batch_in_training_mode_matches_dropout_array_layer_after_the_same_seed(seed: int):
    rng = random.Random(seed)
    numpy_layer, rust_layer = _training_layers(rng)
    _seed_both(seed)
    for _ in range(3):
        x_data = random_matrix(rng, BATCH_SIZE, INPUT_SIZE)
        expected = numpy_layer.forward_batch(np.array(x_data))
        actual = rust_layer.forward_batch(Array(x_data))
        assert rust_to_numpy(rust_layer._mask_batch).tolist() == numpy_layer._mask_batch.tolist()
        assert rust_to_numpy(rust_layer._base_activation_batch) == approx(numpy_layer._base_activation_batch)
        assert rust_to_numpy(actual) == approx(expected)


# Any: a layer, its next layer and a batch of one backend, which a union can't express
def _learn_batch_step(layer: Any, next_layer: Any, X: Any) -> None:
    layer.forward_batch(X)
    layer.compute_hidden_delta_batch(next_layer)
    layer.accumulate_gradient_batch(X)
    layer.apply_accumulated_gradient(0.5, BATCH_SIZE)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_learn_batch_step_in_training_mode_matches_dropout_array_layer_after_the_same_seed(seed: int):
    # forward_batch, hidden delta from a next layer, gradient accumulation and the update
    rng = random.Random(seed)
    numpy_layer, rust_layer = _training_layers(rng)
    next_w_data = random_matrix(rng, NEXT_SIZE, HIDDEN_SIZE)
    next_delta_batch_data = random_matrix(rng, BATCH_SIZE, NEXT_SIZE)
    numpy_next = ArrayLayer(NEXT_SIZE, HIDDEN_SIZE)
    numpy_next.W, numpy_next.delta_batch = np.array(next_w_data), np.array(next_delta_batch_data)
    rust_next = RustArrayLayer(NEXT_SIZE, HIDDEN_SIZE)
    rust_next.W, rust_next.delta_batch = Array(next_w_data), Array(next_delta_batch_data)

    _seed_both(seed)
    for _ in range(3):
        x_data = random_matrix(rng, BATCH_SIZE, INPUT_SIZE)
        _learn_batch_step(numpy_layer, numpy_next, np.array(x_data))
        _learn_batch_step(rust_layer, rust_next, Array(x_data))

        assert rust_to_numpy(rust_layer._mask_batch).tolist() == numpy_layer._mask_batch.tolist()
        assert rust_to_numpy(rust_layer.delta_batch) == approx(numpy_layer.delta_batch)
        assert rust_to_numpy(rust_layer.W) == approx(numpy_layer.W)
        assert rust_to_numpy(rust_layer.b) == approx(numpy_layer.b)


def test_layer_dropout_forward_batch_in_training_mode_draws_an_independent_mask_per_row():
    rng = random.Random(0)
    w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = random_vector(rng, HIDDEN_SIZE)
    x_data = random_matrix(rng, 30, INPUT_SIZE)

    _a, mask, _base = layer_dropout_forward_batch(Array(w_data), Array(x_data), Array(b_data), 0.5, True)
    rows = [tuple(mask[r, c] for c in range(HIDDEN_SIZE)) for r in range(30)]
    assert len(set(rows)) > 1


def test_layer_dropout_hidden_delta_in_training_mode_uses_the_returned_mask_and_base_activation():
    # sidesteps the RNG entirely by forcing mask/base_activation directly - checks the chain-rule
    # formula, not any particular random outcome
    next_w = Array([[0.8]])
    next_delta = Array([-0.5])
    base_activation = Array([0.7502601055951177])

    kept = layer_dropout_hidden_delta(next_w, next_delta, base_activation, Array([1.0]), KEEP_PROBABILITY, True)
    dropped = layer_dropout_hidden_delta(next_w, next_delta, base_activation, Array([0.0]), KEEP_PROBABILITY, True)
    eval_mode = layer_dropout_hidden_delta(next_w, next_delta, base_activation, Array([0.0]), KEEP_PROBABILITY, False)

    base = 0.7502601055951177
    sigmoid_derivative = base * (1.0 - base)
    expected_kept = (-0.5 * 0.8) * sigmoid_derivative / KEEP_PROBABILITY
    expected_eval = (-0.5 * 0.8) * sigmoid_derivative  # eval-mode ignores mask entirely

    assert kept[0] == approx(expected_kept)
    assert dropped[0] == 0.0
    assert eval_mode[0] == approx(expected_eval)


def test_layer_dropout_hidden_delta_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        layer_dropout_hidden_delta(Array.zeros((2, 3)), Array.zeros(2), Array.zeros(5), Array.zeros(5), 0.5, True)
