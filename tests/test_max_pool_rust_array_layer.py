import numpy as np
import pytest

import indrajala_math_rust as pa
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer
from indrajala_ml.model.max_pool_array_layer import MaxPoolArrayLayer
from indrajala_ml.model.max_pool_rust_array_layer import MaxPoolRustArrayLayer
from tests.test_max_pool_array_layer import SHAPES, _tie_heavy_inputs


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


@pytest.mark.parametrize("shape", SHAPES)
def test_geometry_attributes_match_max_pool_array_layer(shape):
    rust_layer, numpy_layer = MaxPoolRustArrayLayer(*shape), MaxPoolArrayLayer(*shape)
    for name in (
        "input_height",
        "input_width",
        "input_channels",
        "pool_size",
        "stride",
        "channel_count",
        "out_height",
        "out_width",
        "input_size",
        "size",
    ):
        assert getattr(rust_layer, name) == getattr(numpy_layer, name), name


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tie_heavy", [False, True])
def test_forward_argmax_and_downstream_match_max_pool_array_layer_single_and_batch(shape, tie_heavy):

    rng = np.random.default_rng(0)
    rust_layer, numpy_layer = MaxPoolRustArrayLayer(*shape), MaxPoolArrayLayer(*shape)
    if tie_heavy:
        X = _tie_heavy_inputs(rng, 5, numpy_layer.input_size)
    else:
        X = rng.uniform(-1.0, 1.0, size=(5, numpy_layer.input_size))
    delta_batch = rng.uniform(-1.0, 1.0, size=(5, numpy_layer.size))

    A = _np(rust_layer.forward_batch(pa.Array(X.tolist())))
    np.testing.assert_array_equal(A, numpy_layer.forward_batch(X))
    argmax_batch = _np(rust_layer.argmax_batch)
    np.testing.assert_array_equal(argmax_batch, numpy_layer.argmax_batch.reshape(5, -1))

    rust_layer.delta_batch = pa.Array(delta_batch.tolist())
    numpy_layer.delta_batch = delta_batch
    downstream_batch = _np(rust_layer.downstream_batch())
    np.testing.assert_array_equal(downstream_batch, numpy_layer.downstream_batch())

    for i, x in enumerate(X):
        a = rust_layer.forward(pa.Array(x.tolist()))
        assert a.shape == rust_layer.argmax.shape == (numpy_layer.size,)
        np.testing.assert_array_equal(_np(a), A[i])
        np.testing.assert_array_equal(_np(rust_layer.argmax), argmax_batch[i])
        rust_layer.delta = pa.Array(delta_batch[i].tolist())
        downstream = _np(rust_layer.downstream())
        assert downstream.shape == (numpy_layer.input_size,)
        np.testing.assert_array_equal(downstream, downstream_batch[i])


@pytest.mark.parametrize(
    ("x", "winner"),
    [
        ([0.0, 0.0, 0.0, 0.0], 0),  # an all-zero window
        ([0.1, 0.9, 0.9, 0.2], 1),  # slots 1 and 2 tie for the max
    ],
)
def test_ties_pick_the_first_slot_as_max_pool_array_layer_does(x, winner):

    rust_layer, numpy_layer = MaxPoolRustArrayLayer(2, 2, 1, 2), MaxPoolArrayLayer(2, 2, 1, 2)
    rust_layer.forward(pa.Array(x))
    numpy_layer.forward(np.array(x))
    assert rust_layer.argmax.tolist() == [float(winner)]
    assert numpy_layer.argmax.ravel().tolist() == [winner]

    rust_layer.delta = pa.Array([0.7])
    expected = [0.0] * 4
    expected[winner] = 0.7
    assert rust_layer.downstream().tolist() == expected


def test_hidden_delta_is_the_downstream_gradient_itself_and_gradient_hooks_are_no_ops():

    layer = MaxPoolRustArrayLayer(4, 4, 1, 2)
    X = pa.Array(np.arange(32, dtype=np.float64).reshape(2, 16).tolist())
    layer.forward_batch(X)
    G = pa.Array(np.arange(8, dtype=np.float64).reshape(2, 4).tolist())
    layer.compute_hidden_delta_batch(_FixedDownstream(G))
    assert layer.delta_batch.tolist() == G.tolist()

    layer.forward(pa.Array(list(range(16))))
    layer.compute_hidden_delta(_FixedDownstream(pa.Array([1.0, 2.0, 3.0, 4.0])))
    assert layer.delta.tolist() == [1.0, 2.0, 3.0, 4.0]

    layer.accumulate_gradient_batch(X)
    layer.accumulate_gradient(pa.Array(list(range(16))))
    layer.apply_accumulated_gradient(0.1, batch_size=2)
    assert not hasattr(layer, "W") and not hasattr(layer, "b")

    with pytest.raises(NotImplementedError):
        layer.compute_output_delta(pa.Array.zeros(4))
    with pytest.raises(NotImplementedError):
        layer.compute_output_delta_batch(pa.Array.zeros((1, 4)))


@pytest.mark.parametrize(
    "arguments",
    [
        (4, 4, 1, 0, None),  # pool_size
        (4, 4, 1, 2, 0),  # stride
        (4, 4, 0, 2, None),  # input_channels
        (4, 3, 1, 4, None),  # window wider than the input
        (3, 4, 1, 4, None),  # window taller than the input
    ],
)
def test_constructor_rejects_invalid_arguments_as_max_pool_array_layer_does(arguments):

    with pytest.raises(AssertionError):
        MaxPoolRustArrayLayer(*arguments)


@pytest.mark.parametrize("pool_stride", [None, 1])
def test_finite_difference_gradients_through_rust_conv_pool_conv(pool_stride):

    # correctness that doesn't rest on agreeing with the numpy layers: for L = sum(G * conv2(pool(
    # conv1(X)))) computed through the Rust layers alone, both kernels' accumulated gradients and
    # conv1's input gradient must match central differences of L. Continuous random inputs keep z
    # off the ReLU kink and keep window maxima distinct within eps.
    rng = np.random.default_rng(1)
    conv1 = ConvRustArrayLayer(8, 8, 2, 3, 3)  # -> 3 x 6 x 6
    pool = MaxPoolRustArrayLayer(6, 6, 3, 2, pool_stride)  # -> 3 x 3 x 3 or 3 x 5 x 5
    conv2 = ConvRustArrayLayer(pool.out_height, pool.out_width, 3, 2, 2)
    params = [
        rng.uniform(-1.0, 1.0, size=(conv1.channel_count, conv1.fan_in)),
        rng.uniform(-0.5, 0.5, size=conv1.channel_count),
        rng.uniform(-1.0, 1.0, size=(conv2.channel_count, conv2.fan_in)),
        rng.uniform(-0.5, 0.5, size=conv2.channel_count),
    ]
    X = rng.uniform(-1.0, 1.0, size=(3, conv1.input_size))
    G = rng.uniform(-1.0, 1.0, size=(3, conv2.size))

    def loss() -> float:
        conv1.W, conv1.b, conv2.W, conv2.b = (pa.Array(p.tolist()) for p in params)
        A = conv2.forward_batch(pool.forward_batch(conv1.forward_batch(pa.Array(X.tolist()))))
        return float(np.sum(G * _np(A)))

    loss()
    conv2.compute_hidden_delta_batch(_FixedDownstream(pa.Array(G.tolist())))
    pool.compute_hidden_delta_batch(conv2)
    conv1.compute_hidden_delta_batch(pool)
    conv1.accumulate_gradient_batch(None)
    conv2.accumulate_gradient_batch(None)
    input_gradient = _np(conv1.downstream_batch())
    analytic = [_np(conv1._grad_W), _np(conv1._grad_b), _np(conv2._grad_W), _np(conv2._grad_b), input_gradient]

    assert np.any(input_gradient != 0.0)

    eps = 1e-6
    for array, gradient in zip([*params, X], analytic):
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + eps
            plus = loss()
            array[index] = original - eps
            minus = loss()
            array[index] = original
            assert gradient[index] == pytest.approx((plus - minus) / (2 * eps), abs=1e-6)
