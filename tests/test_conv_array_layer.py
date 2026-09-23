import numpy as np
import pytest

from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_layer import ConvLayer
from indrajala_ml.model.state_layer import StateLayer

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


class _FixedDownstream:
    """A stand-in next layer whose downstream gradient is fixed - what an upstream layer's
    compute_hidden_delta* reads from whatever follows it."""

    def __init__(self, gradient_batch: np.ndarray) -> None:
        self.gradient_batch = gradient_batch

    def downstream_batch(self) -> np.ndarray:
        return self.gradient_batch

    def downstream(self) -> np.ndarray:
        return self.gradient_batch[0]


def _random_layers(rng, shape) -> tuple[ConvArrayLayer, ConvLayer, StateLayer]:
    height, width, channels, kernel_size, channel_count, stride = shape
    array_layer = ConvArrayLayer(height, width, channels, kernel_size, channel_count, stride)
    array_layer.W = rng.uniform(-1.0, 1.0, size=array_layer.W.shape)
    array_layer.b = rng.uniform(-0.5, 0.5, size=array_layer.b.shape)

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
        kernel.weights = list(array_layer.W[c])
        kernel.bias = float(array_layer.b[c])
    return array_layer, conv_layer, state_layer


def _python_forward(conv_layer: ConvLayer, state_layer: StateLayer, x: np.ndarray) -> list[float]:
    state_layer.update_state(tuple(float(v) for v in x))
    conv_layer.forward()
    return [unit.value() for unit in conv_layer.nodes]


@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_conv_layer_single_and_batch(shape):

    rng = np.random.default_rng(0)
    array_layer, conv_layer, state_layer = _random_layers(rng, shape)
    assert array_layer.size == len(conv_layer.nodes)
    assert (array_layer.out_height, array_layer.out_width) == (conv_layer.out_height, conv_layer.out_width)

    X = rng.uniform(0.0, 1.0, size=(6, array_layer.input_size))
    A = array_layer.forward_batch(X)
    assert A.shape == (6, array_layer.size)

    for i, x in enumerate(X):
        expected = _python_forward(conv_layer, state_layer, x)
        np.testing.assert_allclose(A[i], expected, rtol=0, atol=1e-12)
        # batch-row independence: row i of forward_batch equals forward on example i alone
        np.testing.assert_allclose(array_layer.forward(x), A[i], rtol=0, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_downstream_matches_conv_layer_downstream_sum_for_every_input(shape):

    rng = np.random.default_rng(1)
    array_layer, conv_layer, _state_layer = _random_layers(rng, shape)

    delta_batch = rng.uniform(-1.0, 1.0, size=(3, array_layer.size))
    array_layer.delta_batch = delta_batch
    downstream_batch = array_layer.downstream_batch()
    assert downstream_batch.shape == (3, array_layer.input_size)

    for row, deltas in enumerate(delta_batch):
        for unit, delta in zip(conv_layer.nodes, deltas):
            unit.delta = float(delta)
        expected = [conv_layer.downstream_sum(i) for i in range(array_layer.input_size)]
        np.testing.assert_allclose(downstream_batch[row], expected, rtol=0, atol=1e-12)

        array_layer.delta = deltas
        np.testing.assert_allclose(array_layer.downstream(), downstream_batch[row], rtol=0, atol=1e-12)


def test_stride_past_the_kernel_leaves_unread_inputs_with_exactly_zero_gradient():

    rng = np.random.default_rng(2)
    array_layer, conv_layer, _state_layer = _random_layers(rng, (6, 6, 2, 2, 2, 3))
    array_layer.delta_batch = rng.uniform(-1.0, 1.0, size=(2, array_layer.size))
    dX = array_layer.downstream_batch().reshape(2, 2, 6, 6)

    unread = [i for i in range(array_layer.input_size) if not conv_layer._fan_out[i]]
    assert unread  # rows/columns 2 and 5 of every channel
    for i in unread:
        assert np.all(dX.reshape(2, -1)[:, i] == 0.0)
    assert np.all(dX[:, :, [2, 5], :] == 0.0)
    assert np.all(dX[:, :, :, [2, 5]] == 0.0)


@pytest.mark.parametrize("shape", SHAPES)
def test_gradient_accumulation_matches_conv_kernel_accumulators_over_a_batch(shape):

    rng = np.random.default_rng(3)
    array_layer, conv_layer, state_layer = _random_layers(rng, shape)

    X = rng.uniform(0.0, 1.0, size=(5, array_layer.input_size))
    delta_batch = rng.uniform(-1.0, 1.0, size=(5, array_layer.size))

    array_layer.forward_batch(X)
    array_layer.delta_batch = delta_batch
    array_layer.accumulate_gradient_batch(X)

    for x, deltas in zip(X, delta_batch):
        _python_forward(conv_layer, state_layer, x)
        for unit, delta in zip(conv_layer.nodes, deltas):
            unit.delta = float(delta)
        conv_layer.accumulate_gradients()

    for c, kernel in enumerate(conv_layer.kernels):
        np.testing.assert_allclose(array_layer._grad_W[c], kernel._weight_gradient_accum, rtol=0, atol=1e-11)
        assert array_layer._grad_b[c] == pytest.approx(kernel._bias_gradient_accum, abs=1e-11)

    # the single-example path, looped over the same batch, accumulates the same totals
    single = ConvArrayLayer(*shape)
    single.W, single.b = array_layer.W.copy(), array_layer.b.copy()
    for x, deltas in zip(X, delta_batch):
        single.forward(x)
        single.delta = deltas
        single.accumulate_gradient(x)
    np.testing.assert_allclose(single._grad_W, array_layer._grad_W, rtol=0, atol=1e-12)
    np.testing.assert_allclose(single._grad_b, array_layer._grad_b, rtol=0, atol=1e-12)

    # applying divides by batch_size and resets, as ConvKernel.apply_accumulated_gradient does
    W_before, b_before = array_layer.W.copy(), array_layer.b.copy()
    grad_W, grad_b = array_layer._grad_W.copy(), array_layer._grad_b.copy()
    array_layer.apply_accumulated_gradient(0.1, batch_size=5)
    conv_layer.apply_accumulated_gradients(0.1, batch_size=5)
    np.testing.assert_allclose(array_layer.W, W_before - 0.1 * grad_W / 5, rtol=0, atol=1e-15)
    np.testing.assert_allclose(array_layer.b, b_before - 0.1 * grad_b / 5, rtol=0, atol=1e-15)
    for c, kernel in enumerate(conv_layer.kernels):
        np.testing.assert_allclose(array_layer.W[c], kernel.weights, rtol=0, atol=1e-12)
    assert not array_layer._grad_W.any() and not array_layer._grad_b.any()


def test_relu_derivative_is_zero_at_exactly_z_equals_zero():

    # one kernel of ones, zero bias: the receptive field over the all-zero top-left patch has
    # z == 0 exactly, so its delta is zero even though the downstream gradient isn't
    layer = ConvArrayLayer(3, 3, 1, 2, 1)
    layer.W = np.ones((1, 4))
    layer.b = np.zeros(1)
    x = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    layer.forward(x)
    np.testing.assert_array_equal(layer.z, [0.0, 2.0, 2.0, 3.0])
    np.testing.assert_array_equal(layer.a, [0.0, 2.0, 2.0, 3.0])  # what ConvRustArrayLayer's test pins

    layer.forward_batch(x[np.newaxis, :])
    layer.compute_hidden_delta_batch(_FixedDownstream(np.full((1, 4), 0.5)))
    np.testing.assert_array_equal(layer.delta_batch, [[0.0, 0.5, 0.5, 0.5]])

    layer.compute_hidden_delta(_FixedDownstream(np.full((1, 4), 0.5)))
    np.testing.assert_array_equal(layer.delta, [0.0, 0.5, 0.5, 0.5])


def test_output_deltas_are_rejected():

    layer = ConvArrayLayer(3, 3, 1, 2, 1)
    with pytest.raises(NotImplementedError):
        layer.compute_output_delta(np.zeros(4))
    with pytest.raises(NotImplementedError):
        layer.compute_output_delta_batch(np.zeros((1, 4)))


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
def test_constructor_rejects_invalid_arguments(arguments):

    with pytest.raises(AssertionError):
        ConvArrayLayer(*arguments)


@pytest.mark.parametrize("shape", SHAPES)
def test_finite_difference_gradients_on_the_numpy_layer_alone(shape):

    # correctness that doesn't rest on agreeing with ConvLayer: for L = sum(G * A) with a fixed
    # upstream gradient G, the accumulated kernel/bias gradients and downstream_batch() (dL/dX)
    # must match central differences of L. Inputs and weights are random, so no z sits on the
    # ReLU kink within eps.
    rng = np.random.default_rng(4)
    layer = ConvArrayLayer(*shape)
    layer.W = rng.uniform(-1.0, 1.0, size=layer.W.shape)
    layer.b = rng.uniform(-0.5, 0.5, size=layer.b.shape)
    X = rng.uniform(-1.0, 1.0, size=(3, layer.input_size))
    G = rng.uniform(-1.0, 1.0, size=(3, layer.size))

    probe = ConvArrayLayer(*shape)
    probe.W, probe.b = layer.W.copy(), layer.b.copy()

    def loss() -> float:
        return float(np.sum(G * probe.forward_batch(X)))

    layer.forward_batch(X)
    layer.compute_hidden_delta_batch(_FixedDownstream(G))
    layer.accumulate_gradient_batch(X)
    input_gradient = layer.downstream_batch()

    eps = 1e-6
    for index in np.ndindex(layer.W.shape):
        original = probe.W[index]
        probe.W[index] = original + eps
        plus = loss()
        probe.W[index] = original - eps
        minus = loss()
        probe.W[index] = original
        assert layer._grad_W[index] == pytest.approx((plus - minus) / (2 * eps), abs=1e-6)

    for c in range(layer.channel_count):
        original = probe.b[c]
        probe.b[c] = original + eps
        plus = loss()
        probe.b[c] = original - eps
        minus = loss()
        probe.b[c] = original
        assert layer._grad_b[c] == pytest.approx((plus - minus) / (2 * eps), abs=1e-6)

    for index in np.ndindex(X.shape):
        original = X[index]
        X[index] = original + eps
        plus = loss()
        X[index] = original - eps
        minus = loss()
        X[index] = original
        assert input_gradient[index] == pytest.approx((plus - minus) / (2 * eps), abs=1e-6)
