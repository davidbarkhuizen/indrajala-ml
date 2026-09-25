"""
conv_forward_batch/conv_downstream_batch/conv_accumulate_gradient_batch, one fused Rust call per
ConvArrayLayer method, and layer_downstream/layer_downstream_batch, one per ArrayLayer.downstream*,
each checked against the numpy method it mirrors - the same treatment test_fused_layer_ops.py
gives the dense ops. The Rust ops reduce through linalg.rs's fixed dot-product grouping, not
numpy's, so values are compared with rtol; the im2col columns are pure copies and must match
exactly.
"""

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.array_layer import ArrayLayer, FloatArray
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from tests.helpers import to_numpy
from tests.test_conv_array_layer import SHAPES, ConvShape

BATCH_SIZE = 4
RTOL = 1e-12
ATOL = 1e-14


def _geometry(layer: ConvArrayLayer) -> "pa.ConvGeometry":
    return pa.ConvGeometry(layer.input_height, layer.input_width, layer.input_channels, layer.kernel_size, layer.stride)


def _rust_forward(layer: ConvArrayLayer, X: FloatArray) -> tuple[pa.Array, pa.Array]:
    return pa.conv_forward_batch(
        pa.Array(layer.W.tolist()), pa.Array(X.tolist()), pa.Array(layer.b.tolist()), _geometry(layer)
    )


def _rust_downstream(layer: ConvArrayLayer) -> FloatArray:
    W, delta_batch = pa.Array(layer.W.tolist()), pa.Array(layer.delta_batch.tolist())
    return to_numpy(pa.conv_downstream_batch(W, delta_batch, _geometry(layer)))


def _random_layer(seed: int, shape: ConvShape) -> tuple[ConvArrayLayer, np.random.Generator]:
    rng = np.random.default_rng(seed)
    layer = ConvArrayLayer(*shape)
    layer.W = rng.uniform(-1.0, 1.0, size=layer.W.shape)
    layer.b = rng.uniform(-0.5, 0.5, size=layer.b.shape)
    return layer, rng


@pytest.mark.parametrize("shape", SHAPES)
def test_conv_forward_batch_matches_conv_array_layer_forward_batch(shape: ConvShape):
    layer, rng = _random_layer(0, shape)
    X = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, layer.input_size))
    layer.forward_batch(X)

    A, cols = _rust_forward(layer, X)

    np.testing.assert_allclose(to_numpy(A), layer.A, rtol=RTOL, atol=ATOL)
    # numpy caches (N, P, C*k*k); Rust the same rows flattened to (N*P, C*k*k)
    np.testing.assert_array_equal(to_numpy(cols), layer._cols.reshape(-1, layer.fan_in))


@pytest.mark.parametrize("shape", SHAPES)
def test_conv_downstream_batch_matches_conv_array_layer_downstream_batch(shape: ConvShape):
    layer, rng = _random_layer(1, shape)
    layer.delta_batch = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, layer.size))

    np.testing.assert_allclose(_rust_downstream(layer), layer.downstream_batch(), rtol=RTOL, atol=ATOL)


def test_conv_downstream_batch_gives_unread_inputs_exactly_zero_gradient_in_both_implementations():
    # 6 wide, k=2, s=3: rows/columns 2 and 5 are outside every receptive field
    layer, rng = _random_layer(2, (6, 6, 2, 2, 2, 3))
    layer.delta_batch = rng.uniform(-1.0, 1.0, size=(2, layer.size))

    for dX in (_rust_downstream(layer), layer.downstream_batch()):
        planes = dX.reshape(2, 2, 6, 6)
        assert np.all(planes[:, :, [2, 5], :] == 0.0)
        assert np.all(planes[:, :, :, [2, 5]] == 0.0)


@pytest.mark.parametrize("shape", SHAPES)
def test_conv_accumulate_gradient_batch_matches_conv_array_layer_accumulate_gradient_batch(shape: ConvShape):
    layer, rng = _random_layer(3, shape)
    X = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, layer.input_size))
    layer.forward_batch(X)
    layer.delta_batch = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, layer.size))
    # non-zero starting accumulators, so the op is checked as an accumulate and not an assign
    layer._grad_W = rng.uniform(-1.0, 1.0, size=layer._grad_W.shape)
    layer._grad_b = rng.uniform(-1.0, 1.0, size=layer._grad_b.shape)
    grad_W0, grad_b0 = layer._grad_W.copy(), layer._grad_b.copy()
    layer.accumulate_gradient_batch(X)

    _A, cols = _rust_forward(layer, X)
    grad_W, grad_b = pa.conv_accumulate_gradient_batch(
        pa.Array(layer.delta_batch.tolist()),
        cols,
        pa.Array(grad_W0.tolist()),
        pa.Array(grad_b0.tolist()),
        _geometry(layer),
    )

    np.testing.assert_allclose(to_numpy(grad_W), layer._grad_W, rtol=RTOL, atol=1e-13)
    np.testing.assert_allclose(to_numpy(grad_b), layer._grad_b, rtol=RTOL, atol=1e-13)


def test_layer_downstream_matches_array_layer_downstream():
    rng = np.random.default_rng(4)
    layer = ArrayLayer(5, 9)
    layer.W = rng.uniform(-1.0, 1.0, size=layer.W.shape)
    layer.delta = rng.uniform(-1.0, 1.0, size=5)
    layer.delta_batch = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, 5))

    W = pa.Array(layer.W.tolist())
    downstream = pa.layer_downstream(W, pa.Array(layer.delta.tolist()))
    np.testing.assert_allclose(to_numpy(downstream), layer.downstream(), rtol=RTOL)
    np.testing.assert_allclose(
        to_numpy(pa.layer_downstream_batch(W, pa.Array(layer.delta_batch.tolist()))),
        layer.downstream_batch(),
        rtol=RTOL,
    )
