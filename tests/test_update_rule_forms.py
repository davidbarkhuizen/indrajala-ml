# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, which strict mode takes for constants)
"""
The update rules follow the literature's form and grouping, in all three implementations, so
their results are comparable with the literature and with each other (README, Update rules).
Goyal et al. 2017 (the paper the batch-size-scaling study tests) write minibatch SGD as eq. (2),
w - lr * (g / B), weight decay as eq. (8), w - lr * (g / B + lambda * w), and momentum as eq. (9),
u = m * u + g / B; w - lr * u, with g summed over the batch. Every apply_accumulated_gradient that
implements one of them is checked against the formula in Python floats, bit for bit: the groupings
differ by an ULP at non-power-of-two batch sizes (6, and 96, the last partial batch of a 60000-row
epoch at B = 128 and 512).
"""

import random
import struct
from collections.abc import Callable, Sequence
from functools import partial

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_kernel import ConvKernel
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer
from indrajala_ml.model.l2_array_layer import L2ArrayLayer
from indrajala_ml.model.l2_regularization_layer import make_l2_node_cls
from indrajala_ml.model.l2_rust_array_layer import L2RustArrayLayer
from indrajala_ml.model.momentum_array_layer import MomentumArrayLayer
from indrajala_ml.model.momentum_conv_layer import make_momentum_kernel_cls
from indrajala_ml.model.momentum_layer import make_momentum_node_cls
from indrajala_ml.model.momentum_rust_array_layer import MomentumRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.state_node import StateNode
from tests.helpers import Wrap

BATCH_SIZES = [1, 6, 96, 4, 128, 512]
SEEDS = range(5)
LEARNING_RATE = 0.1
L2_LAMBDA = 0.01
# a dense layer's (size, input_size), and a conv layer's (channel_count, fan_in) for 5 x 5 x 2
# inputs and 3 x 3 kernels
DENSE_SHAPE = (4, 7)
CONV_SHAPE = (3, 18)

Matrix = list[list[float]]
# a step's new (W, b)
Weights = tuple[Sequence[Sequence[float]], Sequence[float]]
# apply(W, b, grad_W, grad_b, batch_size) -> the new (W, b)
Apply = Callable[[Matrix, list[float], Matrix, list[float], int], Weights]
# step(grad_W, grad_b, learning_rate, batch_size) -> the new (W, b)
Step = Callable[[Matrix, list[float], float, int], Weights]
# start(momentum, W, b) -> step
Start = Callable[[float, Matrix, list[float]], Step]
ArrayLayers = ArrayLayer | RustArrayLayer | ConvArrayLayer | ConvRustArrayLayer


def _sgd(w: float, g: float, batch_size: int) -> float:
    return w - LEARNING_RATE * (g / batch_size)


def _weight_decay(w: float, g: float, batch_size: int) -> float:
    return w - LEARNING_RATE * (g / batch_size + L2_LAMBDA * w)


def _nodes(node_cls: type[BackpropNode], W: Matrix, b: list[float]) -> list[BackpropNode]:
    return [
        node_cls(input_nodes=[StateNode() for _ in weights], input_node_weights=list(weights), bias=bias)
        for weights, bias in zip(W, b)
    ]


def _step_nodes(
    nodes: list[BackpropNode], grad_W: Matrix, grad_b: list[float], learning_rate: float, batch_size: int
) -> Weights:
    for node, accum, bias_accum in zip(nodes, grad_W, grad_b):
        node._weight_gradient_accum, node._bias_gradient_accum = list(accum), bias_accum
        node.apply_accumulated_gradient(learning_rate, batch_size)
    return [node.input_node_weights for node in nodes], [node.bias for node in nodes]


def _apply_nodes(
    node_cls: type[BackpropNode], W: Matrix, b: list[float], grad_W: Matrix, grad_b: list[float], batch_size: int
) -> Weights:
    return _step_nodes(_nodes(node_cls, W, b), grad_W, grad_b, LEARNING_RATE, batch_size)


def _kernels(kernel_cls: type[ConvKernel], W: Matrix, b: list[float]) -> list[ConvKernel]:
    # CONV_SHAPE's kernels: 3 x 3 over 2 input channels
    return [kernel_cls(kernel_size=3, in_channels=2, weights=list(weights), bias=bias) for weights, bias in zip(W, b)]


def _step_kernels(
    kernels: list[ConvKernel], grad_W: Matrix, grad_b: list[float], learning_rate: float, batch_size: int
) -> Weights:
    for kernel, accum, bias_accum in zip(kernels, grad_W, grad_b):
        kernel._weight_gradient_accum, kernel._bias_gradient_accum = list(accum), bias_accum
        kernel.apply_accumulated_gradient(learning_rate, batch_size)
    return [kernel.weights for kernel in kernels], [kernel.bias for kernel in kernels]


def _apply_kernels(W: Matrix, b: list[float], grad_W: Matrix, grad_b: list[float], batch_size: int) -> Weights:
    return _step_kernels(_kernels(ConvKernel, W, b), grad_W, grad_b, LEARNING_RATE, batch_size)


def _step_layer(
    layer: ArrayLayers, to_array: Wrap, grad_W: Matrix, grad_b: list[float], learning_rate: float, batch_size: int
) -> Weights:
    layer._grad_W, layer._grad_b = to_array(grad_W), to_array(grad_b)
    layer.apply_accumulated_gradient(learning_rate, batch_size)
    return layer.W.tolist(), layer.b.tolist()


def _apply_layer(
    layer: ArrayLayers,
    to_array: Wrap,
    W: Matrix,
    b: list[float],
    grad_W: Matrix,
    grad_b: list[float],
    batch_size: int,
) -> Weights:
    layer.W, layer.b = to_array(W), to_array(b)
    return _step_layer(layer, to_array, grad_W, grad_b, LEARNING_RATE, batch_size)


def _fresh_layer(make_layer: Callable[[], ArrayLayers], to_array: Wrap) -> Apply:
    # a new layer for every call, so no state carries over between parametrized runs
    return lambda W, b, grad_W, grad_b, batch_size: _apply_layer(
        make_layer(), to_array, W, b, grad_W, grad_b, batch_size
    )


# (name, the formula it follows, the weights' shape, apply(W, b, grad_W, grad_b, batch_size))
IMPLEMENTATIONS: list[tuple[str, Callable[[float, float, int], float], tuple[int, int], Apply]] = [
    ("BackpropNode", _sgd, DENSE_SHAPE, partial(_apply_nodes, BackpropNode)),
    ("ConvKernel", _sgd, CONV_SHAPE, _apply_kernels),
    ("ArrayLayer", _sgd, DENSE_SHAPE, _fresh_layer(lambda: ArrayLayer(*DENSE_SHAPE), np.array)),
    ("ConvArrayLayer", _sgd, CONV_SHAPE, _fresh_layer(lambda: ConvArrayLayer(5, 5, 2, 3, 3), np.array)),
    ("RustArrayLayer", _sgd, DENSE_SHAPE, _fresh_layer(lambda: RustArrayLayer(*DENSE_SHAPE), pa.Array)),
    ("ConvRustArrayLayer", _sgd, CONV_SHAPE, _fresh_layer(lambda: ConvRustArrayLayer(5, 5, 2, 3, 3), pa.Array)),
    ("L2RegularizedBackpropNode", _weight_decay, DENSE_SHAPE, partial(_apply_nodes, make_l2_node_cls(L2_LAMBDA))),
    (
        "L2ArrayLayer",
        _weight_decay,
        DENSE_SHAPE,
        _fresh_layer(lambda: L2ArrayLayer(*DENSE_SHAPE, L2_LAMBDA), np.array),
    ),
    (
        "L2RustArrayLayer",
        _weight_decay,
        DENSE_SHAPE,
        _fresh_layer(lambda: L2RustArrayLayer(*DENSE_SHAPE, L2_LAMBDA), pa.Array),
    ),
]


def _bits(values: Sequence[float]) -> list[bytes]:
    return [struct.pack("<d", v) for v in values]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("name, weight_rule, shape, apply", IMPLEMENTATIONS, ids=[i[0] for i in IMPLEMENTATIONS])
def test_apply_accumulated_gradient_is_the_papers_form_exactly(
    name: str,
    weight_rule: Callable[[float, float, int], float],
    shape: tuple[int, int],
    apply: Apply,
    batch_size: int,
    seed: int,
):
    rng = random.Random(seed)
    rows, cols = shape
    # parameters on the scale of the update: against weights much larger than it, the groupings'
    # ULP difference mostly rounds away in the subtraction (about 1 value in 100 differs, not 1 in 6)
    scale = LEARNING_RATE / batch_size
    W = [[scale * rng.uniform(-3.0, 3.0) for _ in range(cols)] for _ in range(rows)]
    b = [scale * rng.uniform(-3.0, 3.0) for _ in range(rows)]
    grad_W = [[rng.uniform(-3.0, 3.0) for _ in range(cols)] for _ in range(rows)]
    grad_b = [rng.uniform(-3.0, 3.0) for _ in range(rows)]

    new_W, new_b = apply(W, b, grad_W, grad_b, batch_size)

    expected_W = [[weight_rule(w, g, batch_size) for w, g in zip(*row)] for row in zip(W, grad_W)]
    # the bias is plain SGD in every rule, unregularized under weight decay
    expected_b = [_sgd(v, g, batch_size) for v, g in zip(b, grad_b)]
    assert [_bits(row) for row in new_W] == [_bits(row) for row in expected_W]
    assert _bits(new_b) == _bits(expected_b)


MOMENTA = [0.0, 0.9]
# the rate changes between steps, as in warmup: there eq. (9) differs from Rumelhart et al.'s
# eq. (10), v = lr * g / B + m * v; w - v, which folds the rate into the velocity
MOMENTUM_LEARNING_RATES = [0.05, 0.1, 0.4]


def _momentum_nodes(momentum: float, W: Matrix, b: list[float]) -> Step:
    nodes = _nodes(make_momentum_node_cls(momentum), W, b)
    return lambda grad_W, grad_b, learning_rate, batch_size: _step_nodes(
        nodes, grad_W, grad_b, learning_rate, batch_size
    )


def _momentum_kernels(momentum: float, W: Matrix, b: list[float]) -> Step:
    kernels = _kernels(make_momentum_kernel_cls(momentum), W, b)
    return lambda grad_W, grad_b, learning_rate, batch_size: _step_kernels(
        kernels, grad_W, grad_b, learning_rate, batch_size
    )


def _momentum_layer(layer_cls: type[MomentumArrayLayer] | type[MomentumRustArrayLayer], to_array: Wrap) -> Start:
    def start(momentum: float, W: Matrix, b: list[float]) -> Step:
        layer = layer_cls(*DENSE_SHAPE, momentum)
        layer.W, layer.b = to_array(W), to_array(b)
        return lambda grad_W, grad_b, learning_rate, batch_size: _step_layer(
            layer, to_array, grad_W, grad_b, learning_rate, batch_size
        )

    return start


# (name, the weights' shape, start(momentum, W, b) -> step(grad_W, grad_b, learning_rate, batch_size))
MOMENTUM_IMPLEMENTATIONS: list[tuple[str, tuple[int, int], Start]] = [
    ("MomentumBackpropNode", DENSE_SHAPE, _momentum_nodes),
    ("MomentumConvKernel", CONV_SHAPE, _momentum_kernels),
    ("MomentumArrayLayer", DENSE_SHAPE, _momentum_layer(MomentumArrayLayer, np.array)),
    ("MomentumRustArrayLayer", DENSE_SHAPE, _momentum_layer(MomentumRustArrayLayer, pa.Array)),
]


def _momentum(
    w: float, u: float, g: float, momentum: float, learning_rate: float, batch_size: int
) -> tuple[float, float]:
    u = momentum * u + g / batch_size
    return w - learning_rate * u, u


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("momentum", MOMENTA)
@pytest.mark.parametrize("name, shape, start", MOMENTUM_IMPLEMENTATIONS, ids=[i[0] for i in MOMENTUM_IMPLEMENTATIONS])
def test_momentum_is_the_papers_form_exactly(
    name: str, shape: tuple[int, int], start: Start, momentum: float, batch_size: int, seed: int
):
    # eq. (9) for W and b alike, over several steps: the velocity only shows a mistake across
    # repeated steps. At momentum 0.0 this is exactly SGD's w - lr * (g / B)
    rng = random.Random(seed)
    rows, cols = shape
    scale = LEARNING_RATE / batch_size
    W = [[scale * rng.uniform(-3.0, 3.0) for _ in range(cols)] for _ in range(rows)]
    b = [scale * rng.uniform(-3.0, 3.0) for _ in range(rows)]
    step = start(momentum, W, b)
    u_W, u_b = [[0.0] * cols for _ in range(rows)], [0.0] * rows

    for learning_rate in MOMENTUM_LEARNING_RATES:
        grad_W = [[rng.uniform(-3.0, 3.0) for _ in range(cols)] for _ in range(rows)]
        grad_b = [rng.uniform(-3.0, 3.0) for _ in range(rows)]
        new_W, new_b = step(grad_W, grad_b, learning_rate, batch_size)

        stepped_W = [
            [_momentum(w, u, g, momentum, learning_rate, batch_size) for w, u, g in zip(*row)]
            for row in zip(W, u_W, grad_W)
        ]
        W, u_W = [[w for w, _ in row] for row in stepped_W], [[u for _, u in row] for row in stepped_W]
        stepped_b = [_momentum(v, u, g, momentum, learning_rate, batch_size) for v, u, g in zip(b, u_b, grad_b)]
        b, u_b = [v for v, _ in stepped_b], [u for _, u in stepped_b]
        assert [_bits(row) for row in new_W] == [_bits(row) for row in W]
        assert _bits(new_b) == _bits(b)
