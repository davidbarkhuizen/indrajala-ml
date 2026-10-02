"""
`layer_adam_apply_accumulated_gradient` is one fused Rust call for the whole Adam (Kingma & Ba,
2014) update rule per parameter, checked
against the numpy optimizer's Adam rule (`indrajala_ml.model.optimizers.numpy_optimizer.NumpyOptimizer`) - the
production reference this function matches - the same treatment `test_fused_layer_ops.py` gives every non-Adam fused
op.
"""

import random

import numpy as np
import pytest
from indrajala_math_rust import Array, layer_adam_apply_accumulated_gradient

from indrajala_ml.model.layers.numpy.array_layer import ArrayLayer
from indrajala_ml.model.specs.update_rules import Adam
from tests.helpers import LayerOptimizer, approx, random_matrix, random_vector, rust_to_numpy

SEEDS = range(30)
INPUT_SIZE = 8
HIDDEN_SIZE = 5
BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", [1, 6])
def test_layer_adam_apply_accumulated_gradient_matches_the_numpy_adam_optimizer_at_step_one(seed: int, batch_size: int):
    rng = random.Random(seed)
    w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = random_vector(rng, HIDDEN_SIZE)
    grad_w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    grad_b_data = random_vector(rng, HIDDEN_SIZE)
    learning_rate = rng.uniform(0.001, 1.0)

    layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    layer.grad_W, layer.grad_b = np.array(grad_w_data), np.array(grad_b_data)
    optimizer = LayerOptimizer(layer, Adam(BETA1, BETA2, EPSILON))
    optimizer.apply(learning_rate, batch_size)
    m_W, v_W, m_b, v_b = optimizer.state

    new_w, new_b, new_m_w, new_v_w, new_m_b, new_v_b = layer_adam_apply_accumulated_gradient(
        Array(w_data),
        Array(b_data),
        Array(grad_w_data),
        Array(grad_b_data),
        Array.zeros((HIDDEN_SIZE, INPUT_SIZE)),
        Array.zeros((HIDDEN_SIZE, INPUT_SIZE)),
        Array.zeros(HIDDEN_SIZE),
        Array.zeros(HIDDEN_SIZE),
        1,
        BETA1,
        BETA2,
        EPSILON,
        learning_rate,
        batch_size,
    )
    assert rust_to_numpy(new_w) == approx(layer.W)
    assert rust_to_numpy(new_b) == approx(layer.b)
    assert rust_to_numpy(new_m_w) == approx(m_W)
    assert rust_to_numpy(new_v_w) == approx(v_W)
    assert rust_to_numpy(new_m_b) == approx(m_b)
    assert rust_to_numpy(new_v_b) == approx(v_b)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_adam_apply_accumulated_gradient_matches_the_numpy_adam_optimizer_across_several_steps(seed: int):
    # m/v/t only actually exercise their accumulation logic across repeated steps, unlike a
    # stateless update where a single comparison at t=1 would do - mirrors
    # test_array_optimizer_adam.py's own reasoning for using a multi-step sweep, not just one call.
    rng = random.Random(seed)
    layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    optimizer = LayerOptimizer(layer, Adam(BETA1, BETA2, EPSILON))
    layer.W = np.array(random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE))
    layer.b = np.array(random_vector(rng, HIDDEN_SIZE))
    learning_rate = rng.uniform(0.001, 1.0)

    m_w = Array.zeros((HIDDEN_SIZE, INPUT_SIZE))
    v_w = Array.zeros((HIDDEN_SIZE, INPUT_SIZE))
    m_b = Array.zeros(HIDDEN_SIZE)
    v_b = Array.zeros(HIDDEN_SIZE)
    w = Array(layer.W.tolist())
    b = Array(layer.b.tolist())

    for t in range(1, 6):
        grad_w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
        grad_b_data = random_vector(rng, HIDDEN_SIZE)

        layer.grad_W = np.array(grad_w_data)
        layer.grad_b = np.array(grad_b_data)
        optimizer.apply(learning_rate, batch_size=1)

        w, b, m_w, v_w, m_b, v_b = layer_adam_apply_accumulated_gradient(
            w,
            b,
            Array(grad_w_data),
            Array(grad_b_data),
            m_w,
            v_w,
            m_b,
            v_b,
            t,
            BETA1,
            BETA2,
            EPSILON,
            learning_rate,
            1,
        )

        assert rust_to_numpy(w) == approx(layer.W)
        assert rust_to_numpy(b) == approx(layer.b)
        assert optimizer.optimizer.t == t


def test_rejects_batch_size_zero():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    with pytest.raises(ValueError):
        layer_adam_apply_accumulated_gradient(w, b, w, b, w, w, b, b, 1, BETA1, BETA2, EPSILON, 0.1, 0)


def test_rejects_t_zero():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    with pytest.raises(ValueError):
        layer_adam_apply_accumulated_gradient(w, b, w, b, w, w, b, b, 0, BETA1, BETA2, EPSILON, 0.1, 1)


def test_rejects_mismatched_shapes():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    wrong_shape_m_w = Array.zeros((3, 2))
    with pytest.raises(ValueError):
        layer_adam_apply_accumulated_gradient(w, b, w, b, wrong_shape_m_w, w, b, b, 1, BETA1, BETA2, EPSILON, 0.1, 1)
