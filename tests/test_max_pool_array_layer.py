import numpy as np
import pytest

from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.max_pool_array_layer import MaxPoolArrayLayer
from indrajala_ml.model.max_pool_layer import MaxPoolLayer
from indrajala_ml.model.state_layer import StateLayer

# (input_height, input_width, input_channels, pool_size, stride) - non-overlapping (stride None ->
# pool_size), overlapping (stride < pool_size), gapped (stride > pool_size), multi-channel,
# non-square
SHAPES = [
    (4, 4, 1, 2, None),
    (6, 6, 2, 2, None),
    (5, 7, 3, 3, None),
    (5, 5, 2, 3, 1),
    (6, 5, 1, 2, 1),
    (7, 7, 2, 2, 3),
]


class _FixedDownstream:
    def __init__(self, gradient_batch: np.ndarray) -> None:
        self.gradient_batch = gradient_batch

    def downstream_batch(self) -> np.ndarray:
        return self.gradient_batch

    def downstream(self) -> np.ndarray:
        return self.gradient_batch[0]


def _layers(shape) -> tuple[MaxPoolArrayLayer, MaxPoolLayer, StateLayer]:
    height, width, channels, pool_size, stride = shape
    dimension = channels * height * width
    state_layer = StateLayer(dimension, [(-100.0, 100.0)] * dimension)
    python_layer = MaxPoolLayer(
        input_layer=state_layer,
        input_height=height,
        input_width=width,
        input_channels=channels,
        pool_size=pool_size,
        stride=stride,
    )
    return MaxPoolArrayLayer(height, width, channels, pool_size, stride), python_layer, state_layer


def _tie_heavy_inputs(rng, count: int, dimension: int) -> np.ndarray:
    # a few discrete values, half of them exact zeros (what a ReLU layer produces) - so exact
    # ties inside a window are the common case, not a rare one
    return rng.choice([0.0, 0.0, 0.0, 0.25, 0.5, 1.0], size=(count, dimension))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tie_heavy", [False, True])
def test_forward_argmax_and_downstream_match_max_pool_layer(shape, tie_heavy):

    rng = np.random.default_rng(0)
    array_layer, python_layer, state_layer = _layers(shape)
    assert array_layer.size == len(python_layer.nodes)
    assert (array_layer.out_height, array_layer.out_width, array_layer.stride) == (
        python_layer.out_height,
        python_layer.out_width,
        python_layer.stride,
    )

    X = _tie_heavy_inputs(rng, 5, array_layer.input_size) if tie_heavy else rng.uniform(-1.0, 1.0, size=(5, array_layer.input_size))
    delta_batch = rng.uniform(-1.0, 1.0, size=(5, array_layer.size))

    A = array_layer.forward_batch(X)
    argmax_batch = array_layer.argmax_batch.copy()  # the single-example forward below overwrites it
    array_layer.delta_batch = delta_batch
    downstream_batch = array_layer.downstream_batch()

    for i, x in enumerate(X):
        state_layer.update_state(tuple(float(v) for v in x))
        python_layer.forward()
        np.testing.assert_array_equal(A[i], [unit.value() for unit in python_layer.nodes])
        np.testing.assert_array_equal(argmax_batch[i].ravel(), [unit.argmax_slot for unit in python_layer.nodes])

        for unit, delta in zip(python_layer.nodes, delta_batch[i]):
            unit.delta = float(delta)
        expected = [python_layer.downstream_sum(j) for j in range(array_layer.input_size)]
        np.testing.assert_allclose(downstream_batch[i], expected, rtol=0, atol=1e-15)

        # the single-example path agrees with its batch row
        np.testing.assert_array_equal(array_layer.forward(x), A[i])
        array_layer.delta = delta_batch[i]
        np.testing.assert_array_equal(array_layer.downstream(), downstream_batch[i])


def test_an_all_zero_window_picks_slot_0_in_both_implementations():

    array_layer, python_layer, state_layer = _layers((2, 2, 1, 2, None))
    x = np.zeros(4)

    array_layer.forward(x)
    state_layer.update_state(tuple(x))
    python_layer.forward()

    assert array_layer.argmax.ravel().tolist() == [0]
    assert python_layer.nodes[0].argmax_slot == 0

    array_layer.delta = np.array([0.7])
    np.testing.assert_array_equal(array_layer.downstream(), [0.7, 0.0, 0.0, 0.0])


def test_a_partial_tie_picks_the_first_tied_slot_in_row_major_order():

    # slots (0,0)=0.1, (0,1)=0.9, (1,0)=0.9, (1,1)=0.2: slots 1 and 2 tie for the max
    array_layer, python_layer, state_layer = _layers((2, 2, 1, 2, None))
    x = np.array([0.1, 0.9, 0.9, 0.2])

    array_layer.forward(x)
    state_layer.update_state(tuple(x))
    python_layer.forward()

    assert array_layer.argmax.ravel().tolist() == [1]
    assert python_layer.nodes[0].argmax_slot == 1

    array_layer.delta = np.array([0.7])
    np.testing.assert_array_equal(array_layer.downstream(), [0.0, 0.7, 0.0, 0.0])


def test_overlapping_windows_send_every_won_delta_to_the_shared_input():

    # stride 1, pool 2 over a 1x3 row... as 2x3: the centre-column max (5.0) wins both windows
    array_layer = MaxPoolArrayLayer(2, 3, 1, 2, stride=1)
    array_layer.forward(np.array([0.0, 5.0, 0.0, 0.0, 1.0, 0.0]))
    array_layer.delta = np.array([0.25, 0.5])
    np.testing.assert_array_equal(array_layer.downstream(), [0.0, 0.75, 0.0, 0.0, 0.0, 0.0])


def test_hidden_delta_is_the_downstream_gradient_itself_and_gradient_hooks_are_no_ops():

    layer = MaxPoolArrayLayer(4, 4, 1, 2)
    X = np.arange(32, dtype=np.float64).reshape(2, 16)
    layer.forward_batch(X)
    G = np.arange(8, dtype=np.float64).reshape(2, 4)
    layer.compute_hidden_delta_batch(_FixedDownstream(G))
    np.testing.assert_array_equal(layer.delta_batch, G)

    layer.forward(X[0])
    layer.compute_hidden_delta(_FixedDownstream(G))
    np.testing.assert_array_equal(layer.delta, G[0])

    layer.accumulate_gradient_batch(X)
    layer.accumulate_gradient(X[0])
    layer.apply_accumulated_gradient(0.1, batch_size=2)
    assert not hasattr(layer, "W") and not hasattr(layer, "b")

    with pytest.raises(NotImplementedError):
        layer.compute_output_delta(np.zeros(4))
    with pytest.raises(NotImplementedError):
        layer.compute_output_delta_batch(np.zeros((1, 4)))


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
def test_constructor_rejects_invalid_arguments(arguments):

    with pytest.raises(AssertionError):
        MaxPoolArrayLayer(*arguments)


@pytest.mark.parametrize("pool_stride", [None, 1])
def test_finite_difference_gradients_through_conv_pool_conv(pool_stride):

    # correctness that doesn't rest on agreeing with MaxPoolLayer: for L = sum(G * conv2(pool(
    # conv1(X)))), both kernels' accumulated gradients and conv1's input gradient must match
    # central differences of L. Continuous random inputs keep z off the ReLU kink and keep
    # window maxima distinct within eps (all-zero post-ReLU windows are flat either way).
    rng = np.random.default_rng(1)
    conv1 = ConvArrayLayer(8, 8, 2, 3, 3)  # -> 3 x 6 x 6
    pool = MaxPoolArrayLayer(6, 6, 3, 2, pool_stride)  # -> 3 x 3 x 3 or 3 x 5 x 5
    conv2 = ConvArrayLayer(pool.out_height, pool.out_width, 3, 2, 2)
    for conv in (conv1, conv2):
        conv.W = rng.uniform(-1.0, 1.0, size=conv.W.shape)
        conv.b = rng.uniform(-0.5, 0.5, size=conv.b.shape)

    X = rng.uniform(-1.0, 1.0, size=(3, conv1.input_size))
    G = rng.uniform(-1.0, 1.0, size=(3, conv2.size))

    def loss() -> float:
        return float(np.sum(G * conv2.forward_batch(pool.forward_batch(conv1.forward_batch(X)))))

    loss()
    conv2.compute_hidden_delta_batch(_FixedDownstream(G))
    pool.compute_hidden_delta_batch(conv2)
    conv1.compute_hidden_delta_batch(pool)
    conv1.accumulate_gradient_batch(X)
    conv2.accumulate_gradient_batch(None)
    input_gradient = conv1.downstream_batch()
    grads = [(conv1.W, conv1._grad_W.copy()), (conv1.b, conv1._grad_b.copy()),
             (conv2.W, conv2._grad_W.copy()), (conv2.b, conv2._grad_b.copy()),
             (X, input_gradient)]  # fmt: skip

    assert np.any(input_gradient != 0.0)

    eps = 1e-6
    for array, analytic in grads:
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + eps
            plus = loss()
            array[index] = original - eps
            minus = loss()
            array[index] = original
            assert analytic[index] == pytest.approx((plus - minus) / (2 * eps), abs=1e-6)
