import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.relu_array_layer import ReLUArrayLayer
from indrajala_ml.model.relu_layer import ReLULayer
from indrajala_ml.model.relu_rust_array_layer import ReLURustArrayLayer
from indrajala_ml.model.state_layer import StateLayer
from tests.helpers import Backend, set_random_node_weights

LayerCls = type[ReLUArrayLayer] | type[ReLURustArrayLayer]
LAYER_CLS: dict[str, LayerCls] = {"numpy": ReLUArrayLayer, "rust": ReLURustArrayLayer}


@pytest.fixture
def layer_cls(backend: Backend) -> LayerCls:
    return LAYER_CLS[backend.name]


def _array_layer_like(backprop_layer: BackpropLayer, backend: Backend):
    array_layer = LAYER_CLS[backend.name](backprop_layer.size, len(backprop_layer.input_layer.nodes))
    snapshot = backprop_layer.snapshot_state()
    array_layer.W = backend.owned([weights for weights, _bias in snapshot])
    array_layer.b = backend.owned([bias for _weights, bias in snapshot])
    return array_layer


def test_forward_matches_relu_node_across_a_random_sweep_including_the_z_equals_zero_boundary(backend: Backend):

    rng = random.Random(41)
    dimension = 5
    size = 4

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        relu_layer = ReLULayer(size, state_layer)

        set_random_node_weights(rng, relu_layer, dimension, 3.0)

        array_layer = _array_layer_like(relu_layer, backend)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        relu_layer.forward()
        expected = [node.value() for node in relu_layer.nodes]

        actual = array_layer.forward(backend.owned(x))
        assert np.allclose(actual.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_forward_is_exactly_zero_at_the_z_equals_zero_boundary(layer_cls: LayerCls, backend: Backend):

    array_layer = layer_cls(1, 1)
    array_layer.W = backend.owned([[1.0]])
    array_layer.b = backend.owned([0.0])

    result = array_layer.forward(backend.owned([0.0]))
    assert result.tolist()[0] == 0.0


def test_forward_batch_matches_per_row_single_example_results_stacked(layer_cls: LayerCls, backend: Backend):

    rng = random.Random(42)
    dimension = 4
    size = 3
    batch_size = 6

    array_layer = layer_cls(size, dimension)
    array_layer.W = backend.owned([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    array_layer.b = backend.owned([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = [[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)]

    expected = [array_layer.forward(backend.owned(x)).tolist() for x in X]

    actual = array_layer.forward_batch(backend.owned(X))
    assert np.allclose(actual.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_compute_hidden_delta_matches_relu_hidden_delta_across_a_random_sweep(layer_cls: LayerCls, backend: Backend):

    rng = random.Random(43)
    hidden_size = 5
    next_size = 4

    for _ in range(100):
        state_layer = StateLayer(hidden_size, [(-10.0, 10.0)] * hidden_size)
        hidden_layer = ReLULayer(hidden_size, state_layer)
        next_layer = BackpropLayer(next_size, hidden_layer)

        for node in next_layer.nodes:
            node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(hidden_size)])
            node.bias = rng.uniform(-3.0, 3.0)
            node.delta = rng.uniform(-5.0, 5.0)

        activations = [rng.uniform(-5.0, 5.0) for _ in range(hidden_size)]
        for node, a in zip(hidden_layer.nodes, activations):
            node._activation = a

        expected: list[float] = []
        for i, node in enumerate(hidden_layer.nodes):
            node.compute_hidden_delta(next_layer.nodes, i)
            expected.append(node.delta)

        array_hidden = layer_cls(hidden_size, hidden_size)
        array_hidden.a = backend.owned(activations)
        # only W and delta are read; Any: paired with a layer of the same backend, which a union can't express
        array_next: Any = _array_layer_like(next_layer, backend)
        array_next.delta = backend.owned([node.delta for node in next_layer.nodes])

        array_hidden.compute_hidden_delta(array_next)
        assert np.allclose(array_hidden.delta.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_compute_hidden_delta_is_exactly_zero_at_the_activation_equals_zero_boundary(
    layer_cls: LayerCls, backend: Backend
):

    array_hidden = layer_cls(1, 1)
    array_hidden.a = backend.owned([0.0])

    array_next: Any = layer_cls(1, 1)  # Any: paired with a layer of the same backend, which a union can't express
    array_next.W = backend.owned([[3.0]])
    array_next.delta = backend.owned([7.0])

    array_hidden.compute_hidden_delta(array_next)
    assert array_hidden.delta.tolist()[0] == 0.0


def test_compute_hidden_delta_batch_matches_per_row_single_example_results_stacked(
    layer_cls: LayerCls, backend: Backend
):

    rng = random.Random(44)
    hidden_size = 5
    next_size = 3
    batch_size = 7

    next_layer: Any = layer_cls(
        next_size, hidden_size
    )  # Any: paired with a layer of the same backend, which a union can't express
    next_layer.W = backend.owned([[rng.uniform(-3.0, 3.0) for _ in range(hidden_size)] for _ in range(next_size)])
    delta_rows = [[rng.uniform(-5.0, 5.0) for _ in range(next_size)] for _ in range(batch_size)]
    next_layer.delta_batch = backend.owned(delta_rows)

    hidden_layer = layer_cls(hidden_size, 1)
    A = [[rng.uniform(-5.0, 5.0) for _ in range(hidden_size)] for _ in range(batch_size)]

    expected: list[list[float]] = []
    for row_index in range(batch_size):
        hidden_layer.a = backend.owned(A[row_index])
        next_layer.delta = backend.owned(delta_rows[row_index])
        hidden_layer.compute_hidden_delta(next_layer)
        expected.append(hidden_layer.delta.tolist())

    hidden_layer.A = backend.owned(A)
    hidden_layer.compute_hidden_delta_batch(next_layer)
    assert np.allclose(hidden_layer.delta_batch.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_raises_not_implemented(layer_cls: LayerCls, backend: Backend):

    array_layer = layer_cls(2, 3)
    with pytest.raises(NotImplementedError):
        array_layer.compute_output_delta(backend.owned([0.0, 1.0]))


def test_compute_output_delta_batch_raises_not_implemented(layer_cls: LayerCls, backend: Backend):

    array_layer = layer_cls(2, 3)
    with pytest.raises(NotImplementedError):
        array_layer.compute_output_delta_batch(backend.owned([[0.0, 1.0]]))


def test_apply_accumulated_gradient_is_inherited_unchanged_from_array_layer(layer_cls: LayerCls, backend: Backend):

    # ReLU changes the forward and backward formulas, not the weight update
    array_layer = layer_cls(2, 2)
    array_layer.W = backend.owned([[1.0, 2.0], [3.0, 4.0]])
    array_layer.b = backend.owned([5.0, 6.0])
    array_layer.delta = backend.owned([1.0, 1.0])
    array_layer.accumulate_gradient(backend.owned([1.0, 1.0]))
    array_layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(array_layer.W.tolist(), [[0.9, 1.9], [2.9, 3.9]])
    assert np.allclose(array_layer.b.tolist(), [4.9, 5.9])
