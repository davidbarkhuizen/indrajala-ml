"""
`layer_momentum_apply_accumulated_gradient` is one fused Rust call for the whole momentum update
rule, checked against the numpy optimizer's Momentum rule
(`indrajala_ml.model.optimizers.numpy_optimizer.NumpyOptimizer`) - the production reference this function
matches - the same treatment `test_adam_fused_layer_ops.py`/
`test_l2_fused_layer_ops.py` give their own fused ops. The optimizer calls the same op on a conv
W, (channel_count, fan_in), so it is also checked against the numpy optimizer on a conv layer.
"""

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from indrajala_math_rust import Array, layer_momentum_apply_accumulated_gradient

from indrajala_ml.model.array_layer import ArrayLayer, FloatArray
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.specs.update_rules import Momentum
from tests.helpers import LayerOptimizer, random_matrix, random_vector, rust_to_numpy

SEEDS = range(30)
INPUT_SIZE = 8
HIDDEN_SIZE = 5
MOMENTUM = 0.5

# a layer, its apply(learning_rate, batch_size), and its velocities [W, b] after an apply
Stepped = tuple[Any, Callable[[float, int], None], Callable[[], list[FloatArray]]]


def _dense() -> Stepped:
    # stepped by the optimizer's Momentum rule
    layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    optimizer = LayerOptimizer(layer, Momentum(MOMENTUM))
    return layer, optimizer.apply, lambda: optimizer.state


def _conv() -> Stepped:
    # over 5 x 5 x 2 inputs with three 3 x 3 kernels: W (3, 18), stepped by the optimizer's
    # Momentum rule
    layer = ConvArrayLayer(5, 5, 2, 3, 3)
    optimizer = LayerOptimizer(layer, Momentum(MOMENTUM))
    return layer, optimizer.apply, lambda: optimizer.state


LAYERS = {"dense": _dense, "conv": _conv}


def _assert_same_bits(arr: Array, expected: FloatArray):
    assert rust_to_numpy(arr).tobytes() == expected.tobytes()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", [1, 6, 96, 4, 128, 512])
@pytest.mark.parametrize("layer_kind", LAYERS)
def test_layer_momentum_apply_accumulated_gradient_matches_the_numpy_layer_exactly(
    layer_kind: str, seed: int, batch_size: int
):
    # bit for bit, over several steps: both are u = m * u + g / B; w - lr * u (Goyal et al. 2017,
    # eq. (9)). The rate changes between steps, as in warmup, where eq. (9) and eq. (10) differ;
    # the velocity only shows a mistake across repeated steps
    rng = random.Random(seed)
    layer, apply, velocities = LAYERS[layer_kind]()
    rows, cols = layer.W.shape
    layer.W = np.array(random_matrix(rng, rows, cols))
    layer.b = np.array(random_vector(rng, rows))

    w = Array(layer.W.tolist())
    b = Array(layer.b.tolist())
    velocity_w = Array.zeros((rows, cols))
    velocity_b = Array.zeros(rows)

    for _ in range(5):
        grad_w_data = random_matrix(rng, rows, cols)
        grad_b_data = random_vector(rng, rows)
        learning_rate = rng.uniform(0.001, 1.0)

        layer.grad_W = np.array(grad_w_data)
        layer.grad_b = np.array(grad_b_data)
        apply(learning_rate, batch_size)

        w, b, velocity_w, velocity_b = layer_momentum_apply_accumulated_gradient(
            w,
            b,
            Array(grad_w_data),
            Array(grad_b_data),
            velocity_w,
            velocity_b,
            MOMENTUM,
            learning_rate,
            batch_size,
        )

        _assert_same_bits(w, layer.W)
        _assert_same_bits(b, layer.b)
        expected_velocity_w, expected_velocity_b = velocities()
        _assert_same_bits(velocity_w, expected_velocity_w)
        _assert_same_bits(velocity_b, expected_velocity_b)


def test_rejects_batch_size_zero():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    with pytest.raises(ValueError):
        layer_momentum_apply_accumulated_gradient(w, b, w, b, w, b, MOMENTUM, 0.1, 0)


def test_rejects_mismatched_shapes():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    wrong_shape_velocity_w = Array.zeros((3, 2))
    with pytest.raises(ValueError):
        layer_momentum_apply_accumulated_gradient(w, b, w, b, wrong_shape_velocity_w, b, MOMENTUM, 0.1, 1)
