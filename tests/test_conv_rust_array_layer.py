import numpy as np
import pytest

import indrajala_math_rust as pa
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer
from tests.test_conv_array_layer import SHAPES

RTOL = 1e-12
ATOL = 1e-13


class _FixedDownstream:
    """A stand-in next layer whose downstream gradient is fixed - batch or single-example,
    whichever shape the caller passes."""

    def __init__(self, gradient) -> None:
        self.gradient = gradient

    def downstream_batch(self):
        return self.gradient

    def downstream(self):
        return self.gradient


def _np(arr) -> np.ndarray:
    return np.array(arr.tolist())


def _matching_layers(seed, shape) -> tuple[ConvRustArrayLayer, ConvArrayLayer, np.random.Generator]:
    rng = np.random.default_rng(seed)
    numpy_layer = ConvArrayLayer(*shape)
    numpy_layer.W = rng.uniform(-1.0, 1.0, size=numpy_layer.W.shape)
    numpy_layer.b = rng.uniform(-0.5, 0.5, size=numpy_layer.b.shape)
    rust_layer = ConvRustArrayLayer(*shape)
    rust_layer.W = pa.Array(numpy_layer.W.tolist())
    rust_layer.b = pa.Array(numpy_layer.b.tolist())
    return rust_layer, numpy_layer, rng


@pytest.mark.parametrize("shape", SHAPES)
def test_geometry_attributes_match_conv_array_layer(shape):
    rust_layer, numpy_layer, _rng = _matching_layers(0, shape)
    for name in (
        "input_height",
        "input_width",
        "input_channels",
        "kernel_size",
        "channel_count",
        "stride",
        "out_height",
        "out_width",
        "positions",
        "fan_in",
        "input_size",
        "size",
    ):
        assert getattr(rust_layer, name) == getattr(numpy_layer, name), name


@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_conv_array_layer_single_and_batch(shape):
    rust_layer, numpy_layer, rng = _matching_layers(1, shape)
    X = rng.uniform(-1.0, 1.0, size=(5, numpy_layer.input_size))

    A = _np(rust_layer.forward_batch(pa.Array(X.tolist())))
    np.testing.assert_allclose(A, numpy_layer.forward_batch(X), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_np(rust_layer.Z), numpy_layer.Z, rtol=RTOL, atol=ATOL)

    for i, x in enumerate(X):
        a = _np(rust_layer.forward(pa.Array(x.tolist())))
        assert rust_layer.a.shape == (numpy_layer.size,)
        # single-example path == the batch row, exactly: the same op on the same row
        np.testing.assert_array_equal(a, A[i])
        np.testing.assert_allclose(a, numpy_layer.forward(x), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(_np(rust_layer.z), numpy_layer.z, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("shape", SHAPES)
def test_hidden_delta_and_downstream_match_conv_array_layer_single_and_batch(shape):
    rust_layer, numpy_layer, rng = _matching_layers(2, shape)
    X = rng.uniform(-1.0, 1.0, size=(4, numpy_layer.input_size))
    G = rng.uniform(-1.0, 1.0, size=(4, numpy_layer.size))

    rust_layer.forward_batch(pa.Array(X.tolist()))
    numpy_layer.forward_batch(X)
    rust_layer.compute_hidden_delta_batch(_FixedDownstream(pa.Array(G.tolist())))
    numpy_layer.compute_hidden_delta_batch(_FixedDownstream(G))
    np.testing.assert_allclose(_np(rust_layer.delta_batch), numpy_layer.delta_batch, rtol=RTOL, atol=ATOL)

    downstream_batch = _np(rust_layer.downstream_batch())
    np.testing.assert_allclose(downstream_batch, numpy_layer.downstream_batch(), rtol=RTOL, atol=ATOL)

    for i, x in enumerate(X):
        rust_layer.forward(pa.Array(x.tolist()))
        numpy_layer.forward(x)
        rust_layer.compute_hidden_delta(_FixedDownstream(pa.Array(G[i].tolist())))
        numpy_layer.compute_hidden_delta(_FixedDownstream(G[i]))
        np.testing.assert_allclose(_np(rust_layer.delta), numpy_layer.delta, rtol=RTOL, atol=ATOL)
        downstream = _np(rust_layer.downstream())
        assert downstream.shape == (numpy_layer.input_size,)
        np.testing.assert_array_equal(downstream, downstream_batch[i])


@pytest.mark.parametrize("shape", SHAPES)
def test_accumulate_and_apply_match_conv_array_layer_single_and_batch(shape):
    rust_layer, numpy_layer, rng = _matching_layers(3, shape)
    X = rng.uniform(-1.0, 1.0, size=(5, numpy_layer.input_size))
    delta_batch = rng.uniform(-1.0, 1.0, size=(5, numpy_layer.size))

    rust_layer.forward_batch(pa.Array(X.tolist()))
    numpy_layer.forward_batch(X)
    rust_layer.delta_batch = pa.Array(delta_batch.tolist())
    numpy_layer.delta_batch = delta_batch
    rust_layer.accumulate_gradient_batch(pa.Array(X.tolist()))
    numpy_layer.accumulate_gradient_batch(X)
    np.testing.assert_allclose(_np(rust_layer._grad_W), numpy_layer._grad_W, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_np(rust_layer._grad_b), numpy_layer._grad_b, rtol=RTOL, atol=ATOL)

    # the single-example path, looped over the same batch, accumulates the same totals
    single = ConvRustArrayLayer(*shape)
    single.W, single.b = rust_layer.W.copy(), rust_layer.b.copy()
    for x, deltas in zip(X, delta_batch):
        single.forward(pa.Array(x.tolist()))
        single.delta = pa.Array(deltas.tolist())
        single.accumulate_gradient(pa.Array(x.tolist()))
    np.testing.assert_allclose(_np(single._grad_W), numpy_layer._grad_W, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_np(single._grad_b), numpy_layer._grad_b, rtol=RTOL, atol=ATOL)

    rust_layer.apply_accumulated_gradient(0.1, batch_size=5)
    numpy_layer.apply_accumulated_gradient(0.1, batch_size=5)
    np.testing.assert_allclose(_np(rust_layer.W), numpy_layer.W, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_np(rust_layer.b), numpy_layer.b, rtol=RTOL, atol=ATOL)
    assert not _np(rust_layer._grad_W).any() and not _np(rust_layer._grad_b).any()
    assert rust_layer._grad_W.shape == (numpy_layer.channel_count, numpy_layer.fan_in)


def test_relu_derivative_is_zero_at_exactly_z_equals_zero():

    # one kernel of ones, zero bias: the receptive field over the all-zero top-left patch has
    # z == 0 exactly, so its delta is zero even though the downstream gradient isn't
    layer = ConvRustArrayLayer(3, 3, 1, 2, 1)
    layer.W = pa.Array([[1.0, 1.0, 1.0, 1.0]])
    layer.b = pa.Array.zeros(1)
    x = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    layer.forward(pa.Array(x))
    assert layer.z.tolist() == [0.0, 2.0, 2.0, 3.0]

    layer.forward_batch(pa.Array([x]))
    layer.compute_hidden_delta_batch(_FixedDownstream(pa.Array([[0.5] * 4])))
    assert layer.delta_batch.tolist() == [[0.0, 0.5, 0.5, 0.5]]

    layer.compute_hidden_delta(_FixedDownstream(pa.Array([0.5] * 4)))
    assert layer.delta.tolist() == [0.0, 0.5, 0.5, 0.5]


def test_output_deltas_are_rejected():

    layer = ConvRustArrayLayer(3, 3, 1, 2, 1)
    with pytest.raises(NotImplementedError):
        layer.compute_output_delta(pa.Array.zeros(4))
    with pytest.raises(NotImplementedError):
        layer.compute_output_delta_batch(pa.Array.zeros((1, 4)))


@pytest.mark.parametrize(
    "arguments",
    [
        (3, 3, 0, 2, 1, 1),  # input_channels
        (3, 3, 1, 0, 1, 1),  # kernel_size
        (3, 3, 1, 2, 0, 1),  # channel_count
        (3, 3, 1, 2, 1, 0),  # stride
        (3, 5, 1, 4, 1, 1),  # kernel taller than the input
        (5, 3, 1, 4, 1, 1),  # kernel wider than the input
    ],
)
def test_constructor_rejects_invalid_arguments_as_conv_array_layer_does(arguments):

    with pytest.raises(AssertionError):
        ConvRustArrayLayer(*arguments)


@pytest.mark.parametrize("shape", SHAPES)
def test_finite_difference_gradients_on_the_rust_layer_alone(shape):

    # correctness that doesn't rest on agreeing with ConvArrayLayer: for L = sum(G * A) with a
    # fixed upstream gradient G, the accumulated kernel/bias gradients and downstream_batch()
    # (dL/dX) must match central differences of L computed through the Rust forward pass alone.
    rng = np.random.default_rng(4)
    layer = ConvRustArrayLayer(*shape)
    W = rng.uniform(-1.0, 1.0, size=(layer.channel_count, layer.fan_in))
    b = rng.uniform(-0.5, 0.5, size=layer.channel_count)
    X = rng.uniform(-1.0, 1.0, size=(3, layer.input_size))
    G = rng.uniform(-1.0, 1.0, size=(3, layer.size))
    layer.W, layer.b = pa.Array(W.tolist()), pa.Array(b.tolist())

    probe = ConvRustArrayLayer(*shape)

    def loss(W, b, X) -> float:
        probe.W, probe.b = pa.Array(W.tolist()), pa.Array(b.tolist())
        return float(np.sum(G * _np(probe.forward_batch(pa.Array(X.tolist())))))

    layer.forward_batch(pa.Array(X.tolist()))
    layer.compute_hidden_delta_batch(_FixedDownstream(pa.Array(G.tolist())))
    layer.accumulate_gradient_batch(pa.Array(X.tolist()))
    grad_W, grad_b = _np(layer._grad_W), _np(layer._grad_b)
    input_gradient = _np(layer.downstream_batch())

    eps = 1e-6
    for array, gradient in ((W, grad_W), (b, grad_b), (X, input_gradient)):
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + eps
            plus = loss(W, b, X)
            array[index] = original - eps
            minus = loss(W, b, X)
            array[index] = original
            assert gradient[index] == pytest.approx((plus - minus) / (2 * eps), abs=1e-6)
