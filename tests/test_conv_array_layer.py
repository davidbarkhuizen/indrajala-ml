import numpy as np
import pytest

from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_layer import ConvLayer
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer
from indrajala_ml.model.state_layer import StateLayer

LAYER_CLS = {"numpy": ConvArrayLayer, "rust": ConvRustArrayLayer}

# (input_height, input_width, input_channels, kernel_size, channel_count, stride) - 1 and 3 input
# channels, stride 1 and 2, square and non-square inputs, and a stride that leaves some inputs
# outside every receptive field (6 wide, k=2, s=3 never reads columns 2 and 5)
SHAPES = [
    (5, 5, 1, 3, 2, 1),
    (6, 6, 1, 2, 3, 2),
    (5, 7, 3, 3, 2, 1),
    (7, 6, 3, 2, 2, 2),
    (6, 6, 2, 2, 2, 3),
    (4, 4, 2, 4, 3, 1),
]

# every backend is within 3e-15 of the pure-Python ConvLayer on these tests
ATOL = 1e-13


class _FixedDownstream:
    """A stand-in next layer whose downstream gradient is fixed: what compute_hidden_delta*
    reads from the layer after it."""

    def __init__(self, gradient_batch, gradient) -> None:
        self.gradient_batch = gradient_batch
        self.gradient = gradient

    def downstream_batch(self):
        return self.gradient_batch

    def downstream(self):
        return self.gradient


def _fixed_downstream(backend, gradient_batch: np.ndarray) -> _FixedDownstream:
    return _FixedDownstream(backend.owned(gradient_batch.tolist()), backend.owned(gradient_batch[0].tolist()))


def _np(array) -> np.ndarray:
    return np.array(array.tolist())


@pytest.fixture
def layer_cls(backend):
    return LAYER_CLS[backend.name]


def _random_layers(rng, shape, backend):
    height, width, channels, kernel_size, channel_count, stride = shape
    array_layer = LAYER_CLS[backend.name](*shape)
    W = rng.uniform(-1.0, 1.0, size=(channel_count, channels * kernel_size**2))
    b = rng.uniform(-0.5, 0.5, size=channel_count)
    array_layer.W, array_layer.b = backend.owned(W.tolist()), backend.owned(b.tolist())

    dimension = channels * height * width
    state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
    conv_layer = ConvLayer(
        input_layer=state_layer,
        input_height=height,
        input_width=width,
        kernel_size=kernel_size,
        channel_count=channel_count,
        stride=stride,
        input_channels=channels,
    )
    for c, kernel in enumerate(conv_layer.kernels):
        kernel.weights = list(W[c])
        kernel.bias = float(b[c])
    return array_layer, conv_layer, state_layer


def _python_forward(conv_layer: ConvLayer, state_layer: StateLayer, x) -> list[float]:
    state_layer.update_state(tuple(float(v) for v in x))
    conv_layer.forward()
    return [unit.value() for unit in conv_layer.nodes]


@pytest.mark.parametrize("shape", SHAPES)
def test_geometry_attributes_follow_from_the_shape(layer_cls, shape):

    height, width, channels, kernel_size, channel_count, stride = shape
    layer = layer_cls(*shape)
    out_height = (height - kernel_size) // stride + 1
    out_width = (width - kernel_size) // stride + 1
    expected = {
        "input_height": height,
        "input_width": width,
        "input_channels": channels,
        "kernel_size": kernel_size,
        "channel_count": channel_count,
        "stride": stride,
        "out_height": out_height,
        "out_width": out_width,
        "positions": out_height * out_width,
        "fan_in": channels * kernel_size**2,
        "input_size": channels * height * width,
        "size": channel_count * out_height * out_width,
    }
    for name, value in expected.items():
        assert getattr(layer, name) == value, name


@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_conv_layer_single_and_batch(backend, shape):

    rng = np.random.default_rng(0)
    array_layer, conv_layer, state_layer = _random_layers(rng, shape, backend)
    assert array_layer.size == len(conv_layer.nodes)
    assert (array_layer.out_height, array_layer.out_width) == (conv_layer.out_height, conv_layer.out_width)

    X = rng.uniform(0.0, 1.0, size=(6, array_layer.input_size))
    A = _np(array_layer.forward_batch(backend.owned(X.tolist())))
    assert A.shape == (6, array_layer.size)

    for i, x in enumerate(X):
        expected = _python_forward(conv_layer, state_layer, x)
        np.testing.assert_allclose(A[i], expected, rtol=0, atol=ATOL)
        # the single-example path is the batch op on one row, so it matches exactly
        a = _np(array_layer.forward(backend.owned(x.tolist())))
        assert a.shape == (array_layer.size,)
        np.testing.assert_array_equal(a, A[i])


@pytest.mark.parametrize("shape", SHAPES)
def test_downstream_matches_conv_layer_downstream_sum_for_every_input(backend, shape):

    rng = np.random.default_rng(1)
    array_layer, conv_layer, _state_layer = _random_layers(rng, shape, backend)

    delta_batch = rng.uniform(-1.0, 1.0, size=(3, array_layer.size))
    array_layer.delta_batch = backend.owned(delta_batch.tolist())
    downstream_batch = _np(array_layer.downstream_batch())
    assert downstream_batch.shape == (3, array_layer.input_size)

    for row, deltas in enumerate(delta_batch):
        for unit, delta in zip(conv_layer.nodes, deltas):
            unit.delta = float(delta)
        expected = [conv_layer.downstream_sum(i) for i in range(array_layer.input_size)]
        np.testing.assert_allclose(downstream_batch[row], expected, rtol=0, atol=ATOL)

        array_layer.delta = backend.owned(deltas.tolist())
        downstream = _np(array_layer.downstream())
        assert downstream.shape == (array_layer.input_size,)
        np.testing.assert_array_equal(downstream, downstream_batch[row])


@pytest.mark.parametrize("shape", SHAPES)
def test_hidden_delta_matches_conv_unit_single_and_batch(backend, shape):

    # the ReLU mask on a fixed downstream gradient, against ConvUnit.compute_hidden_delta
    rng = np.random.default_rng(2)
    array_layer, conv_layer, state_layer = _random_layers(rng, shape, backend)
    X = rng.uniform(-1.0, 1.0, size=(4, array_layer.input_size))
    G = rng.uniform(-1.0, 1.0, size=(4, array_layer.size))

    array_layer.forward_batch(backend.owned(X.tolist()))
    array_layer.compute_hidden_delta_batch(_fixed_downstream(backend, G))
    delta_batch = _np(array_layer.delta_batch)

    for i, x in enumerate(X):
        _python_forward(conv_layer, state_layer, x)
        for unit, g in zip(conv_layer.nodes, G[i]):
            unit.compute_hidden_delta(float(g))
        np.testing.assert_allclose(delta_batch[i], [unit.delta for unit in conv_layer.nodes], rtol=0, atol=ATOL)

        array_layer.forward(backend.owned(x.tolist()))
        array_layer.compute_hidden_delta(_fixed_downstream(backend, G[i : i + 1]))
        np.testing.assert_array_equal(_np(array_layer.delta), delta_batch[i])


def test_stride_past_the_kernel_leaves_unread_inputs_with_exactly_zero_gradient(backend):

    rng = np.random.default_rng(2)
    array_layer, conv_layer, _state_layer = _random_layers(rng, (6, 6, 2, 2, 2, 3), backend)
    array_layer.delta_batch = backend.owned(rng.uniform(-1.0, 1.0, size=(2, array_layer.size)).tolist())
    dX = _np(array_layer.downstream_batch()).reshape(2, 2, 6, 6)

    unread = [i for i in range(array_layer.input_size) if not conv_layer._fan_out[i]]
    assert unread  # rows/columns 2 and 5 of every channel
    for i in unread:
        assert np.all(dX.reshape(2, -1)[:, i] == 0.0)
    assert np.all(dX[:, :, [2, 5], :] == 0.0)
    assert np.all(dX[:, :, :, [2, 5]] == 0.0)


@pytest.mark.parametrize("shape", SHAPES)
def test_gradient_accumulation_matches_conv_kernel_accumulators_over_a_batch(backend, layer_cls, shape):

    rng = np.random.default_rng(3)
    array_layer, conv_layer, state_layer = _random_layers(rng, shape, backend)

    X = rng.uniform(0.0, 1.0, size=(5, array_layer.input_size))
    delta_batch = rng.uniform(-1.0, 1.0, size=(5, array_layer.size))

    array_layer.forward_batch(backend.owned(X.tolist()))
    array_layer.delta_batch = backend.owned(delta_batch.tolist())
    array_layer.accumulate_gradient_batch(backend.owned(X.tolist()))
    grad_W, grad_b = _np(array_layer._grad_W), _np(array_layer._grad_b)

    for x, deltas in zip(X, delta_batch):
        _python_forward(conv_layer, state_layer, x)
        for unit, delta in zip(conv_layer.nodes, deltas):
            unit.delta = float(delta)
        conv_layer.accumulate_gradients()

    for c, kernel in enumerate(conv_layer.kernels):
        np.testing.assert_allclose(grad_W[c], kernel._weight_gradient_accum, rtol=0, atol=ATOL)
        assert grad_b[c] == pytest.approx(kernel._bias_gradient_accum, rel=0, abs=ATOL)

    # the single-example path, looped over the same batch, accumulates the same totals
    single = layer_cls(*shape)
    single.W, single.b = array_layer.W.copy(), array_layer.b.copy()
    for x, deltas in zip(X, delta_batch):
        single.forward(backend.owned(x.tolist()))
        single.delta = backend.owned(deltas.tolist())
        single.accumulate_gradient(backend.owned(x.tolist()))
    np.testing.assert_allclose(_np(single._grad_W), grad_W, rtol=0, atol=ATOL)
    np.testing.assert_allclose(_np(single._grad_b), grad_b, rtol=0, atol=ATOL)

    # applying divides by batch_size and resets, as ConvKernel.apply_accumulated_gradient does
    W_before, b_before = _np(array_layer.W), _np(array_layer.b)
    array_layer.apply_accumulated_gradient(0.1, batch_size=5)
    conv_layer.apply_accumulated_gradients(0.1, batch_size=5)
    np.testing.assert_allclose(_np(array_layer.W), W_before - 0.1 * grad_W / 5, rtol=0, atol=1e-15)
    np.testing.assert_allclose(_np(array_layer.b), b_before - 0.1 * grad_b / 5, rtol=0, atol=1e-15)
    for c, kernel in enumerate(conv_layer.kernels):
        np.testing.assert_allclose(_np(array_layer.W)[c], kernel.weights, rtol=0, atol=ATOL)
    assert not _np(array_layer._grad_W).any() and not _np(array_layer._grad_b).any()
    assert _np(array_layer._grad_W).shape == (array_layer.channel_count, array_layer.fan_in)


def test_relu_derivative_is_zero_at_exactly_z_equals_zero(backend, layer_cls):

    # one kernel of ones, zero bias: the receptive field over the all-zero top-left patch has
    # z == 0 exactly, so its delta is zero even though the downstream gradient isn't
    layer = layer_cls(3, 3, 1, 2, 1)
    layer.W = backend.owned([[1.0, 1.0, 1.0, 1.0]])
    layer.b = backend.owned([0.0])
    x = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    layer.forward(backend.owned(x))
    assert layer.a.tolist() == [0.0, 2.0, 2.0, 3.0]
    if backend.name == "numpy":
        assert layer.z.tolist() == [0.0, 2.0, 2.0, 3.0]
    else:
        # the Rust op applies the ReLU as it writes A and keeps no z
        assert not hasattr(layer, "z") and not hasattr(layer, "Z")

    layer.forward_batch(backend.owned([x]))
    layer.compute_hidden_delta_batch(_fixed_downstream(backend, np.full((1, 4), 0.5)))
    assert layer.delta_batch.tolist() == [[0.0, 0.5, 0.5, 0.5]]

    layer.compute_hidden_delta(_fixed_downstream(backend, np.full((1, 4), 0.5)))
    assert layer.delta.tolist() == [0.0, 0.5, 0.5, 0.5]


def test_output_deltas_are_rejected(backend, layer_cls):

    layer = layer_cls(3, 3, 1, 2, 1)
    with pytest.raises(NotImplementedError):
        layer.compute_output_delta(backend.owned([0.0] * 4))
    with pytest.raises(NotImplementedError):
        layer.compute_output_delta_batch(backend.owned([[0.0] * 4]))


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
def test_constructor_rejects_invalid_arguments(layer_cls, arguments):

    with pytest.raises(AssertionError):
        layer_cls(*arguments)


@pytest.mark.parametrize("shape", SHAPES)
def test_finite_difference_gradients_on_the_layer_alone(backend, layer_cls, shape):

    # correctness that doesn't rest on agreeing with another implementation: for L = sum(G * A)
    # with a fixed upstream gradient G, the accumulated kernel/bias gradients and
    # downstream_batch() (dL/dX) must match central differences of L through this layer's own
    # forward pass. Inputs and weights are random, so no z sits on the ReLU kink within eps.
    rng = np.random.default_rng(4)
    layer = layer_cls(*shape)
    W = rng.uniform(-1.0, 1.0, size=(layer.channel_count, layer.fan_in))
    b = rng.uniform(-0.5, 0.5, size=layer.channel_count)
    X = rng.uniform(-1.0, 1.0, size=(3, layer.input_size))
    G = rng.uniform(-1.0, 1.0, size=(3, layer.size))
    layer.W, layer.b = backend.owned(W.tolist()), backend.owned(b.tolist())

    probe = layer_cls(*shape)

    def loss() -> float:
        probe.W, probe.b = backend.owned(W.tolist()), backend.owned(b.tolist())
        return float(np.sum(G * _np(probe.forward_batch(backend.owned(X.tolist())))))

    layer.forward_batch(backend.owned(X.tolist()))
    layer.compute_hidden_delta_batch(_fixed_downstream(backend, G))
    layer.accumulate_gradient_batch(backend.owned(X.tolist()))
    grad_W, grad_b = _np(layer._grad_W), _np(layer._grad_b)
    input_gradient = _np(layer.downstream_batch())

    eps = 1e-6
    for array, gradient in ((W, grad_W), (b, grad_b), (X, input_gradient)):
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + eps
            plus = loss()
            array[index] = original - eps
            minus = loss()
            array[index] = original
            assert gradient[index] == pytest.approx((plus - minus) / (2 * eps), abs=1e-6)
