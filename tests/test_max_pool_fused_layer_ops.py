# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, which strict mode takes for constants)
"""
max_pool_forward_batch/max_pool_downstream_batch, one fused Rust call per MaxPoolArrayLayer method,
checked against it over test_max_pool_array_layer.SHAPES with continuous and tie-heavy inputs.
Pooling has no matmul in it: the forward pass picks values, and the downstream scatter-add sums
in numpy's slot order. So everything must match exactly, not within a tolerance - argmax most of
all, since a different tie winner would route a gradient to a different input.
"""

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.max_pool_array_layer import MaxPoolArrayLayer
from tests.helpers import to_numpy
from tests.test_max_pool_array_layer import SHAPES, PoolShape, _tie_heavy_inputs

BATCH_SIZE = 5


def _geometry(layer: MaxPoolArrayLayer) -> "pa.ConvGeometry":
    return pa.ConvGeometry(layer.input_height, layer.input_width, layer.input_channels, layer.pool_size, layer.stride)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tie_heavy", [False, True])
def test_max_pool_ops_match_max_pool_array_layer_exactly(shape: PoolShape, tie_heavy: bool):
    rng = np.random.default_rng(0)
    layer = MaxPoolArrayLayer(*shape)
    if tie_heavy:
        X = _tie_heavy_inputs(rng, BATCH_SIZE, layer.input_size)
    else:
        X = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, layer.input_size))
    layer.forward_batch(X)
    layer.delta_batch = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, layer.size))

    A, argmax = pa.max_pool_forward_batch(pa.Array(X.tolist()), _geometry(layer))
    np.testing.assert_array_equal(to_numpy(A), layer.A)
    np.testing.assert_array_equal(to_numpy(argmax), layer.argmax_batch.reshape(BATCH_SIZE, -1))

    dX = pa.max_pool_downstream_batch(pa.Array(layer.delta_batch.tolist()), argmax, _geometry(layer))
    np.testing.assert_array_equal(to_numpy(dX), layer.downstream_batch())
