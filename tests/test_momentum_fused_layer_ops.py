"""
`layer_momentum_apply_accumulated_gradient` is one fused Rust call for the whole momentum update
rule, checked against
`indrajala_ml.model.momentum_array_layer.MomentumArrayLayer` - the actual production reference
this function replaces - the same treatment `test_adam_fused_layer_ops.py`/
`test_l2_fused_layer_ops.py` give their own fused ops. The conv layers call the same op on a conv
W, (channel_count, fan_in), so it is also checked against `MomentumConvArrayLayer`.
"""

import random

import numpy as np
import pytest
from indrajala_math_rust import Array, layer_momentum_apply_accumulated_gradient

from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.momentum_array_layer import MomentumArrayLayer
from indrajala_ml.model.momentum_conv_array_layer import MomentumConvArrayLayer
from tests.helpers import random_matrix, random_vector, rust_to_numpy

SEEDS = range(30)
INPUT_SIZE = 8
HIDDEN_SIZE = 5
MOMENTUM = 0.5

# a dense layer, and a conv layer over 5 x 5 x 2 inputs with three 3 x 3 kernels: W (3, 18)
LAYERS = {
    "dense": lambda: MomentumArrayLayer(HIDDEN_SIZE, INPUT_SIZE, MOMENTUM),
    "conv": lambda: MomentumConvArrayLayer(5, 5, 2, 3, 3, momentum=MOMENTUM),
}


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
    layer = LAYERS[layer_kind]()
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

        layer._grad_W = np.array(grad_w_data)
        layer._grad_b = np.array(grad_b_data)
        layer.apply_accumulated_gradient(learning_rate, batch_size)

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
        _assert_same_bits(velocity_w, layer._velocity_W)
        _assert_same_bits(velocity_b, layer._velocity_b)


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
