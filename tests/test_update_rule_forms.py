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
from indrajala_ml.model.momentum_layer import make_momentum_node_cls
from indrajala_ml.model.momentum_rust_array_layer import MomentumRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.state_node import StateNode

BATCH_SIZES = [1, 6, 96, 4, 128, 512]
SEEDS = range(5)
LEARNING_RATE = 0.1
L2_LAMBDA = 0.01
# a dense layer's (size, input_size), and a conv layer's (channel_count, fan_in) for 5 x 5 x 2
# inputs and 3 x 3 kernels
DENSE_SHAPE = (4, 7)
CONV_SHAPE = (3, 18)


def _sgd(w, g, batch_size):
    return w - LEARNING_RATE * (g / batch_size)


def _weight_decay(w, g, batch_size):
    return w - LEARNING_RATE * (g / batch_size + L2_LAMBDA * w)


def _nodes(node_cls, W, b):
    return [
        node_cls(input_nodes=[StateNode() for _ in weights], input_node_weights=list(weights), bias=bias)
        for weights, bias in zip(W, b)
    ]


def _step_nodes(nodes, grad_W, grad_b, learning_rate, batch_size):
    for node, accum, bias_accum in zip(nodes, grad_W, grad_b):
        node._weight_gradient_accum, node._bias_gradient_accum = list(accum), bias_accum
        node.apply_accumulated_gradient(learning_rate, batch_size)
    return [node.input_node_weights for node in nodes], [node.bias for node in nodes]


def _apply_nodes(node_cls, W, b, grad_W, grad_b, batch_size):
    return _step_nodes(_nodes(node_cls, W, b), grad_W, grad_b, LEARNING_RATE, batch_size)


def _apply_kernels(W, b, grad_W, grad_b, batch_size):
    kernels = []
    for weights, bias, accum, bias_accum in zip(W, b, grad_W, grad_b):
        kernel = ConvKernel(kernel_size=3, in_channels=2, weights=list(weights), bias=bias)
        kernel._weight_gradient_accum, kernel._bias_gradient_accum = list(accum), bias_accum
        kernel.apply_accumulated_gradient(LEARNING_RATE, batch_size)
        kernels.append(kernel)
    return [kernel.weights for kernel in kernels], [kernel.bias for kernel in kernels]


def _step_layer(layer, to_array, grad_W, grad_b, learning_rate, batch_size):
    layer._grad_W, layer._grad_b = to_array(grad_W), to_array(grad_b)
    layer.apply_accumulated_gradient(learning_rate, batch_size)
    return layer.W.tolist(), layer.b.tolist()


def _apply_layer(layer, to_array, W, b, grad_W, grad_b, batch_size):
    layer.W, layer.b = to_array(W), to_array(b)
    return _step_layer(layer, to_array, grad_W, grad_b, LEARNING_RATE, batch_size)


def _numpy(layer):
    return lambda *args: _apply_layer(layer, np.array, *args)


def _rust(layer):
    return lambda *args: _apply_layer(layer, pa.Array, *args)


# (name, the formula it follows, the weights' shape, apply(W, b, grad_W, grad_b, batch_size))
IMPLEMENTATIONS = [
    ("BackpropNode", _sgd, DENSE_SHAPE, lambda *args: _apply_nodes(BackpropNode, *args)),
    ("ConvKernel", _sgd, CONV_SHAPE, _apply_kernels),
    ("ArrayLayer", _sgd, DENSE_SHAPE, lambda *args: _numpy(ArrayLayer(*DENSE_SHAPE))(*args)),
    ("ConvArrayLayer", _sgd, CONV_SHAPE, lambda *args: _numpy(ConvArrayLayer(5, 5, 2, 3, 3))(*args)),
    ("RustArrayLayer", _sgd, DENSE_SHAPE, lambda *args: _rust(RustArrayLayer(*DENSE_SHAPE))(*args)),
    ("ConvRustArrayLayer", _sgd, CONV_SHAPE, lambda *args: _rust(ConvRustArrayLayer(5, 5, 2, 3, 3))(*args)),
    (
        "L2RegularizedBackpropNode",
        _weight_decay,
        DENSE_SHAPE,
        lambda *args: _apply_nodes(make_l2_node_cls(L2_LAMBDA), *args),
    ),
    ("L2ArrayLayer", _weight_decay, DENSE_SHAPE, lambda *args: _numpy(L2ArrayLayer(*DENSE_SHAPE, L2_LAMBDA))(*args)),
    (
        "L2RustArrayLayer",
        _weight_decay,
        DENSE_SHAPE,
        lambda *args: _rust(L2RustArrayLayer(*DENSE_SHAPE, L2_LAMBDA))(*args),
    ),
]


def _bits(values):
    return [struct.pack("<d", v) for v in values]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("name, weight_rule, shape, apply", IMPLEMENTATIONS, ids=[i[0] for i in IMPLEMENTATIONS])
def test_apply_accumulated_gradient_is_the_papers_form_exactly(name, weight_rule, shape, apply, batch_size, seed):
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


def _momentum_nodes(momentum, W, b):
    nodes = _nodes(make_momentum_node_cls(momentum), W, b)
    return lambda *args: _step_nodes(nodes, *args)


def _momentum_layer(layer_cls, to_array):
    def start(momentum, W, b):
        layer = layer_cls(*DENSE_SHAPE, momentum)
        layer.W, layer.b = to_array(W), to_array(b)
        return lambda *args: _step_layer(layer, to_array, *args)

    return start


# (name, start(momentum, W, b) -> step(grad_W, grad_b, learning_rate, batch_size))
MOMENTUM_IMPLEMENTATIONS = [
    ("MomentumBackpropNode", _momentum_nodes),
    ("MomentumArrayLayer", _momentum_layer(MomentumArrayLayer, np.array)),
    ("MomentumRustArrayLayer", _momentum_layer(MomentumRustArrayLayer, pa.Array)),
]


def _momentum(w, u, g, momentum, learning_rate, batch_size):
    u = momentum * u + g / batch_size
    return w - learning_rate * u, u


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("momentum", MOMENTA)
@pytest.mark.parametrize("name, start", MOMENTUM_IMPLEMENTATIONS, ids=[i[0] for i in MOMENTUM_IMPLEMENTATIONS])
def test_momentum_is_the_papers_form_exactly(name, start, momentum, batch_size, seed):
    # eq. (9) for W and b alike, over several steps: the velocity only shows a mistake across
    # repeated steps. At momentum 0.0 this is exactly SGD's w - lr * (g / B)
    rng = random.Random(seed)
    rows, cols = DENSE_SHAPE
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
