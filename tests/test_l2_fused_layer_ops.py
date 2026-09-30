"""
`layer_l2_apply_accumulated_gradient` is one fused Rust call for the whole L2 (weight decay)
update rule, checked against the numpy optimizer's WeightDecay rule
(`indrajala_ml.model.optimizers.NumpyOptimizer`) - the production reference this function
matches - the same treatment `test_adam_fused_layer_ops.py` gives Adam's own fused op.
"""

import random

import numpy as np
import pytest
from indrajala_math_rust import Array, layer_l2_apply_accumulated_gradient

from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.update_rules import WeightDecay
from tests.helpers import LayerOptimizer, random_matrix, random_vector, rust_to_numpy

SEEDS = range(30)
INPUT_SIZE = 8
HIDDEN_SIZE = 5
L2_LAMBDA = 0.05


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", [1, 6, 96, 4, 128, 512])
def test_layer_l2_apply_accumulated_gradient_matches_the_numpy_weight_decay_optimizer_exactly(
    seed: int, batch_size: int
):
    # bit for bit: both are w - lr * (g / B + l2_lambda * w) (test_update_rule_forms.py)
    rng = random.Random(seed)
    w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = random_vector(rng, HIDDEN_SIZE)
    grad_w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    grad_b_data = random_vector(rng, HIDDEN_SIZE)
    learning_rate = rng.uniform(0.001, 1.0)

    layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    layer.grad_W, layer.grad_b = np.array(grad_w_data), np.array(grad_b_data)
    LayerOptimizer(layer, WeightDecay(L2_LAMBDA)).apply(learning_rate, batch_size)

    new_w, new_b = layer_l2_apply_accumulated_gradient(
        Array(w_data),
        Array(b_data),
        Array(grad_w_data),
        Array(grad_b_data),
        L2_LAMBDA,
        learning_rate,
        batch_size,
    )
    assert rust_to_numpy(new_w).tobytes() == layer.W.tobytes()
    assert rust_to_numpy(new_b).tobytes() == layer.b.tobytes()


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_l2_apply_accumulated_gradient_matches_exactly_across_several_steps(seed: int):
    rng = random.Random(seed)
    layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    optimizer = LayerOptimizer(layer, WeightDecay(L2_LAMBDA))
    layer.W = np.array(random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE))
    layer.b = np.array(random_vector(rng, HIDDEN_SIZE))
    learning_rate = rng.uniform(0.001, 1.0)

    w = Array(layer.W.tolist())
    b = Array(layer.b.tolist())

    for _ in range(5):
        grad_w_data = random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
        grad_b_data = random_vector(rng, HIDDEN_SIZE)

        layer.grad_W = np.array(grad_w_data)
        layer.grad_b = np.array(grad_b_data)
        optimizer.apply(learning_rate, batch_size=1)

        w, b = layer_l2_apply_accumulated_gradient(
            w, b, Array(grad_w_data), Array(grad_b_data), L2_LAMBDA, learning_rate, 1
        )

        assert rust_to_numpy(w).tobytes() == layer.W.tobytes()
        assert rust_to_numpy(b).tobytes() == layer.b.tobytes()


def test_rejects_batch_size_zero():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    with pytest.raises(ValueError):
        layer_l2_apply_accumulated_gradient(w, b, w, b, L2_LAMBDA, 0.1, 0)


def test_rejects_mismatched_shapes():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    wrong_shape_grad_w = Array.zeros((3, 2))
    with pytest.raises(ValueError):
        layer_l2_apply_accumulated_gradient(w, b, wrong_shape_grad_w, b, L2_LAMBDA, 0.1, 1)
