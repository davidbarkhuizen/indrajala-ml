import random

import numpy as np
import pytest

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.l2_array_layer import L2ArrayLayer
from indrajala_ml.model.l2_regularization_layer import make_l2_layer_cls
from indrajala_ml.model.l2_rust_array_layer import L2RustArrayLayer
from indrajala_ml.model.state_layer import StateLayer
from tests.helpers import Backend

L2_LAMBDA = 0.05

LayerCls = type[L2ArrayLayer] | type[L2RustArrayLayer]
LAYER_CLS: dict[str, LayerCls] = {"numpy": L2ArrayLayer, "rust": L2RustArrayLayer}


@pytest.fixture
def layer_cls(backend: Backend) -> LayerCls:
    return LAYER_CLS[backend.name]


def _array_layer_like(backprop_layer: BackpropLayer, backend: Backend):
    input_size = len(backprop_layer.input_layer.nodes)
    array_layer = LAYER_CLS[backend.name](backprop_layer.size, input_size, L2_LAMBDA)
    snapshot = backprop_layer.snapshot_state()
    array_layer.W = backend.owned([weights for weights, _bias in snapshot])
    array_layer.b = backend.owned([bias for _weights, bias in snapshot])
    return array_layer


def test_accumulate_then_apply_at_batch_size_one_matches_l2_backprop_node_at_every_step(backend: Backend):

    # compared after every step: the penalty depends on the current W, so a mistake can show
    # only once W has moved
    rng = random.Random(21)
    dimension = 5
    size = 4
    node_layer_cls = make_l2_layer_cls(L2_LAMBDA)

    state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
    backprop_layer = node_layer_cls(size=size, input_layer=state_layer)
    for node in backprop_layer.nodes:
        node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
        node.bias = rng.uniform(-3.0, 3.0)

    array_layer = _array_layer_like(backprop_layer, backend)
    learning_rate = rng.uniform(0.001, 1.0)

    for _ in range(10):
        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        deltas = [rng.uniform(-5.0, 5.0) for _ in range(size)]

        for node, delta in zip(backprop_layer.nodes, deltas):
            node.delta = delta
            node.accumulate_gradient()
            node.apply_accumulated_gradient(learning_rate, batch_size=1)

        array_layer.delta = backend.owned(deltas)
        array_layer.accumulate_gradient(backend.owned(x))
        array_layer.apply_accumulated_gradient(learning_rate, batch_size=1)

        expected_W = np.array([node.input_node_weights for node in backprop_layer.nodes])
        expected_b = np.array([node.bias for node in backprop_layer.nodes])
        assert np.allclose(array_layer.W.tolist(), expected_W, rtol=1e-9, atol=1e-12)
        assert np.allclose(array_layer.b.tolist(), expected_b, rtol=1e-9, atol=1e-12)


def test_accumulate_across_a_batch_then_apply_matches_l2_backprop_node_at_every_batch(backend: Backend):

    rng = random.Random(22)
    dimension = 4
    size = 3
    batch_size = 6
    node_layer_cls = make_l2_layer_cls(L2_LAMBDA)

    state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
    backprop_layer = node_layer_cls(size=size, input_layer=state_layer)
    for node in backprop_layer.nodes:
        node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
        node.bias = rng.uniform(-3.0, 3.0)

    array_layer = _array_layer_like(backprop_layer, backend)
    learning_rate = rng.uniform(0.001, 1.0)

    for _ in range(5):
        examples = [
            ([rng.uniform(-10.0, 10.0) for _ in range(dimension)], [rng.uniform(-5.0, 5.0) for _ in range(size)])
            for _ in range(batch_size)
        ]

        for x, deltas in examples:
            state_layer.update_state(tuple(x))
            for node, delta in zip(backprop_layer.nodes, deltas):
                node.delta = delta
                node.accumulate_gradient()

            array_layer.delta = backend.owned(deltas)
            array_layer.accumulate_gradient(backend.owned(x))

        for node in backprop_layer.nodes:
            node.apply_accumulated_gradient(learning_rate, batch_size)
        array_layer.apply_accumulated_gradient(learning_rate, batch_size)

        expected_W = np.array([node.input_node_weights for node in backprop_layer.nodes])
        expected_b = np.array([node.bias for node in backprop_layer.nodes])
        assert np.allclose(array_layer.W.tolist(), expected_W, rtol=1e-9, atol=1e-12)
        assert np.allclose(array_layer.b.tolist(), expected_b, rtol=1e-9, atol=1e-12)


def test_bias_is_never_regularized(layer_cls: LayerCls, backend: Backend):

    # with a zero weight gradient and a nonzero bias gradient, the bias moves by plain SGD: the
    # penalty applies to W only
    array_layer = layer_cls(2, 2, l2_lambda=0.5)
    array_layer.W = backend.owned([[1.0, 2.0], [3.0, 4.0]])
    array_layer.b = backend.owned([5.0, 6.0])
    array_layer.delta = backend.owned([0.0, 0.0])
    array_layer.accumulate_gradient(backend.owned([0.0, 0.0]))
    array_layer._grad_b = backend.owned([2.0, 4.0])

    array_layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(array_layer.b.tolist(), [5.0 - 0.1 * 2.0, 6.0 - 0.1 * 4.0])


def test_apply_accumulated_gradient_resets_the_accumulator(layer_cls: LayerCls, backend: Backend):

    array_layer = layer_cls(3, 2, L2_LAMBDA)
    array_layer.delta = backend.owned([0.1, 0.2, 0.3])
    array_layer.accumulate_gradient(backend.owned([1.0, 2.0]))
    array_layer.apply_accumulated_gradient(0.1, batch_size=1)

    assert np.allclose(array_layer._grad_W.tolist(), np.zeros((3, 2)))
    assert np.allclose(array_layer._grad_b.tolist(), np.zeros(3))
