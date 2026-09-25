import numpy as np
import pytest

from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer
from indrajala_ml.model.max_pool_array_layer import MaxPoolArrayLayer
from indrajala_ml.model.max_pool_layer import MaxPoolLayer
from indrajala_ml.model.max_pool_rust_array_layer import MaxPoolRustArrayLayer
from indrajala_ml.model.state_layer import StateLayer

LAYER_CLS = {"numpy": MaxPoolArrayLayer, "rust": MaxPoolRustArrayLayer}
CONV_LAYER_CLS = {"numpy": ConvArrayLayer, "rust": ConvRustArrayLayer}

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


def _layers(shape, backend):
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
    return LAYER_CLS[backend.name](height, width, channels, pool_size, stride), python_layer, state_layer


def _tie_heavy_inputs(rng, count: int, dimension: int) -> np.ndarray:
    # a few discrete values, half of them exact zeros (what a ReLU layer produces) - so exact
    # ties inside a window are the common case, not a rare one
    return rng.choice([0.0, 0.0, 0.0, 0.25, 0.5, 1.0], size=(count, dimension))


@pytest.mark.parametrize("shape", SHAPES)
def test_geometry_attributes_follow_from_the_shape(layer_cls, shape):

    height, width, channels, pool_size, stride = shape
    layer = layer_cls(*shape)
    effective_stride = pool_size if stride is None else stride
    out_height = (height - pool_size) // effective_stride + 1
    out_width = (width - pool_size) // effective_stride + 1
    expected = {
        "input_height": height,
        "input_width": width,
        "input_channels": channels,
        "pool_size": pool_size,
        "stride": effective_stride,
        "channel_count": channels,
        "out_height": out_height,
        "out_width": out_width,
        "input_size": channels * height * width,
        "size": channels * out_height * out_width,
    }
    for name, value in expected.items():
        assert getattr(layer, name) == value, name


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tie_heavy", [False, True])
def test_forward_argmax_and_downstream_match_max_pool_layer(backend, shape, tie_heavy):

    rng = np.random.default_rng(0)
    array_layer, python_layer, state_layer = _layers(shape, backend)
    assert array_layer.size == len(python_layer.nodes)
    assert (array_layer.out_height, array_layer.out_width, array_layer.stride) == (
        python_layer.out_height,
        python_layer.out_width,
        python_layer.stride,
    )

    if tie_heavy:
        X = _tie_heavy_inputs(rng, 5, array_layer.input_size)
    else:
        X = rng.uniform(-1.0, 1.0, size=(5, array_layer.input_size))
    delta_batch = rng.uniform(-1.0, 1.0, size=(5, array_layer.size))

    A = _np(array_layer.forward_batch(backend.owned(X.tolist())))
    # copied: the single-example forward below overwrites it. Slot indices; Rust stores floats
    argmax_batch = _np(array_layer.argmax_batch).reshape(5, -1)
    array_layer.delta_batch = backend.owned(delta_batch.tolist())
    downstream_batch = _np(array_layer.downstream_batch())

    for i, x in enumerate(X):
        state_layer.update_state(tuple(float(v) for v in x))
        python_layer.forward()
        np.testing.assert_array_equal(A[i], [unit.value() for unit in python_layer.nodes])
        np.testing.assert_array_equal(argmax_batch[i], [unit.argmax_slot for unit in python_layer.nodes])

        for unit, delta in zip(python_layer.nodes, delta_batch[i]):
            unit.delta = float(delta)
        expected = [python_layer.downstream_sum(j) for j in range(array_layer.input_size)]
        np.testing.assert_allclose(downstream_batch[i], expected, rtol=0, atol=1e-15)

        # the single-example path agrees exactly with its batch row
        a = _np(array_layer.forward(backend.owned(x.tolist())))
        assert a.shape == (array_layer.size,)
        np.testing.assert_array_equal(a, A[i])
        np.testing.assert_array_equal(_np(array_layer.argmax).ravel(), argmax_batch[i])
        array_layer.delta = backend.owned(delta_batch[i].tolist())
        downstream = _np(array_layer.downstream())
        assert downstream.shape == (array_layer.input_size,)
        np.testing.assert_array_equal(downstream, downstream_batch[i])


def test_an_all_zero_window_picks_slot_0_in_both_implementations(backend):

    array_layer, python_layer, state_layer = _layers((2, 2, 1, 2, None), backend)
    x = [0.0, 0.0, 0.0, 0.0]

    array_layer.forward(backend.owned(x))
    state_layer.update_state(tuple(x))
    python_layer.forward()

    assert _np(array_layer.argmax).ravel().tolist() == [0]
    assert python_layer.nodes[0].argmax_slot == 0

    array_layer.delta = backend.owned([0.7])
    assert array_layer.downstream().tolist() == [0.7, 0.0, 0.0, 0.0]


def test_a_partial_tie_picks_the_first_tied_slot_in_row_major_order(backend):

    # slots (0,0)=0.1, (0,1)=0.9, (1,0)=0.9, (1,1)=0.2: slots 1 and 2 tie for the max
    array_layer, python_layer, state_layer = _layers((2, 2, 1, 2, None), backend)
    x = [0.1, 0.9, 0.9, 0.2]

    array_layer.forward(backend.owned(x))
    state_layer.update_state(tuple(x))
    python_layer.forward()

    assert _np(array_layer.argmax).ravel().tolist() == [1]
    assert python_layer.nodes[0].argmax_slot == 1

    array_layer.delta = backend.owned([0.7])
    assert array_layer.downstream().tolist() == [0.0, 0.7, 0.0, 0.0]


def test_overlapping_windows_send_every_won_delta_to_the_shared_input(backend, layer_cls):

    # stride 1, pool 2 over a 2x3 input: the centre-column max (5.0) wins both windows
    array_layer = layer_cls(2, 3, 1, 2, stride=1)
    array_layer.forward(backend.owned([0.0, 5.0, 0.0, 0.0, 1.0, 0.0]))
    array_layer.delta = backend.owned([0.25, 0.5])
    assert array_layer.downstream().tolist() == [0.0, 0.75, 0.0, 0.0, 0.0, 0.0]


def test_hidden_delta_is_the_downstream_gradient_itself_and_gradient_hooks_are_no_ops(backend, layer_cls):

    layer = layer_cls(4, 4, 1, 2)
    X = np.arange(32, dtype=np.float64).reshape(2, 16)
    layer.forward_batch(backend.owned(X.tolist()))
    G = np.arange(8, dtype=np.float64).reshape(2, 4)
    layer.compute_hidden_delta_batch(_fixed_downstream(backend, G))
    assert layer.delta_batch.tolist() == G.tolist()

    layer.forward(backend.owned(X[0].tolist()))
    layer.compute_hidden_delta(_fixed_downstream(backend, G))
    assert layer.delta.tolist() == G[0].tolist()

    layer.accumulate_gradient_batch(backend.owned(X.tolist()))
    layer.accumulate_gradient(backend.owned(X[0].tolist()))
    layer.apply_accumulated_gradient(0.1, batch_size=2)
    assert not hasattr(layer, "W") and not hasattr(layer, "b")

    with pytest.raises(NotImplementedError):
        layer.compute_output_delta(backend.owned([0.0] * 4))
    with pytest.raises(NotImplementedError):
        layer.compute_output_delta_batch(backend.owned([[0.0] * 4]))


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
def test_constructor_rejects_invalid_arguments(layer_cls, arguments):

    with pytest.raises(AssertionError):
        layer_cls(*arguments)


@pytest.mark.parametrize("pool_stride", [None, 1])
def test_finite_difference_gradients_through_conv_pool_conv(backend, layer_cls, pool_stride):

    # correctness that doesn't rest on agreeing with MaxPoolLayer: for L = sum(G * conv2(pool(
    # conv1(X)))), both kernels' accumulated gradients and conv1's input gradient must match
    # central differences of L. Continuous random inputs keep z off the ReLU kink and keep
    # window maxima distinct within eps (all-zero post-ReLU windows are flat either way).
    rng = np.random.default_rng(1)
    conv_cls = CONV_LAYER_CLS[backend.name]
    conv1 = conv_cls(8, 8, 2, 3, 3)  # -> 3 x 6 x 6
    pool = layer_cls(6, 6, 3, 2, pool_stride)  # -> 3 x 3 x 3 or 3 x 5 x 5
    conv2 = conv_cls(pool.out_height, pool.out_width, 3, 2, 2)
    params = [
        rng.uniform(-1.0, 1.0, size=(conv1.channel_count, conv1.fan_in)),
        rng.uniform(-0.5, 0.5, size=conv1.channel_count),
        rng.uniform(-1.0, 1.0, size=(conv2.channel_count, conv2.fan_in)),
        rng.uniform(-0.5, 0.5, size=conv2.channel_count),
    ]
    X = rng.uniform(-1.0, 1.0, size=(3, conv1.input_size))
    G = rng.uniform(-1.0, 1.0, size=(3, conv2.size))

    def loss() -> float:
        conv1.W, conv1.b, conv2.W, conv2.b = (backend.owned(p.tolist()) for p in params)
        A = conv2.forward_batch(pool.forward_batch(conv1.forward_batch(backend.owned(X.tolist()))))
        return float(np.sum(G * _np(A)))

    loss()
    conv2.compute_hidden_delta_batch(_fixed_downstream(backend, G))
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
