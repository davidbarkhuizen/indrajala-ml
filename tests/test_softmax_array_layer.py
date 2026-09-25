import random

import numpy as np
import pytest

from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.softmax_array_layer import SoftmaxArrayLayer
from indrajala_ml.model.softmax_output_layer import SoftmaxOutputLayer
from indrajala_ml.model.softmax_rust_array_layer import SoftmaxRustArrayLayer
from indrajala_ml.model.state_layer import StateLayer
from tests.helpers import set_random_node_weights

LAYER_CLS = {"numpy": SoftmaxArrayLayer, "rust": SoftmaxRustArrayLayer}
BASE_LAYER_CLS = {"numpy": ArrayLayer, "rust": RustArrayLayer}


@pytest.fixture
def layer_cls(backend):
    return LAYER_CLS[backend.name]


def _array_layer_like(softmax_layer: SoftmaxOutputLayer, backend):
    array_layer = LAYER_CLS[backend.name](softmax_layer.size, len(softmax_layer.input_layer.nodes))
    snapshot = softmax_layer.snapshot_state()
    array_layer.W = backend.owned([weights for weights, _bias in snapshot])
    array_layer.b = backend.owned([bias for _weights, bias in snapshot])
    return array_layer


def test_forward_matches_softmax_output_layer_across_a_random_sweep(backend):

    rng = random.Random(50)
    dimension = 4
    size = 5

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        softmax_layer = SoftmaxOutputLayer(size, state_layer)

        set_random_node_weights(rng, softmax_layer, dimension, 3.0)

        array_layer = _array_layer_like(softmax_layer, backend)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        softmax_layer.forward()
        expected = [node.value() for node in softmax_layer.nodes]

        actual = array_layer.forward(backend.owned(x)).tolist()
        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
        assert sum(actual) == pytest.approx(1.0)


def test_forward_matches_softmax_output_layer_for_large_magnitude_z_without_overflow(backend):

    # the max shift keeps e^z finite, as the per-node reference's does
    dimension = 1
    state_layer = StateLayer(dimension, [(-1.0, 1.0)])
    softmax_layer = SoftmaxOutputLayer(3, state_layer)
    for node, (weight, bias) in zip(softmax_layer.nodes, [(1.0, 10_000.0), (1.0, 0.0), (1.0, -10_000.0)]):
        node.update_input_weights([weight])
        node.bias = bias
    state_layer.update_state((0.0,))

    array_layer = _array_layer_like(softmax_layer, backend)

    softmax_layer.forward()
    expected = [node.value() for node in softmax_layer.nodes]

    actual = array_layer.forward(backend.owned([0.0])).tolist()
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
    assert actual[0] == pytest.approx(1.0)
    assert actual[1] == pytest.approx(0.0)
    assert actual[2] == pytest.approx(0.0)
    assert not np.isnan(actual).any()


def test_forward_batch_matches_per_row_single_example_results_stacked(layer_cls, backend):

    rng = random.Random(51)
    dimension = 4
    size = 3
    batch_size = 6

    array_layer = layer_cls(size, dimension)
    array_layer.W = backend.owned([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    array_layer.b = backend.owned([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = [[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)]

    expected = [array_layer.forward(backend.owned(x)).tolist() for x in X]

    actual = np.array(array_layer.forward_batch(backend.owned(X)).tolist())
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
    assert np.allclose(actual.sum(axis=1), 1.0)


def test_compute_output_delta_matches_softmax_output_node_across_a_random_sweep(backend):

    rng = random.Random(52)
    dimension = 4
    size = 5

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        softmax_layer = SoftmaxOutputLayer(size, state_layer)

        for node in softmax_layer.nodes:
            node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
            node.bias = rng.uniform(-3.0, 3.0)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        softmax_layer.forward()

        category = rng.randrange(size)
        expected = []
        for i, node in enumerate(softmax_layer.nodes):
            target = 1.0 if i == category else 0.0
            node.compute_output_delta(target)
            expected.append(node.delta)

        array_layer = _array_layer_like(softmax_layer, backend)
        array_layer.forward(backend.owned(x))
        array_layer.compute_output_delta(backend.owned([1.0 if i == category else 0.0 for i in range(size)]))

        assert np.allclose(array_layer.delta.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_batch_matches_per_row_single_example_results_stacked(layer_cls, backend):

    rng = random.Random(53)
    dimension = 4
    size = 3
    batch_size = 5

    array_layer = layer_cls(size, dimension)
    array_layer.W = backend.owned([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    array_layer.b = backend.owned([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = [[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)]
    categories = [rng.randrange(size) for _ in range(batch_size)]
    reference_rows = [[1.0 if i == category else 0.0 for i in range(size)] for category in categories]

    expected = []
    for row in range(batch_size):
        array_layer.forward(backend.owned(X[row]))
        array_layer.compute_output_delta(backend.owned(reference_rows[row]))
        expected.append(array_layer.delta.tolist())

    array_layer.forward_batch(backend.owned(X))
    array_layer.compute_output_delta_batch(backend.owned(reference_rows))
    assert np.allclose(array_layer.delta_batch.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_compute_hidden_delta_is_inherited_unchanged_from_array_layer(layer_cls, backend):

    # softmax couples the nodes in the forward pass only; the layer before it uses the plain
    # hidden-delta formula
    base = BASE_LAYER_CLS[backend.name]
    assert layer_cls.__mro__[1] is base
    assert layer_cls.compute_hidden_delta is base.compute_hidden_delta
    assert layer_cls.compute_hidden_delta_batch is base.compute_hidden_delta_batch


def test_apply_accumulated_gradient_is_inherited_unchanged_from_array_layer(layer_cls, backend):

    array_layer = layer_cls(2, 2)
    array_layer.W = backend.owned([[1.0, 2.0], [3.0, 4.0]])
    array_layer.b = backend.owned([5.0, 6.0])
    array_layer.delta = backend.owned([1.0, 1.0])
    array_layer.accumulate_gradient(backend.owned([1.0, 1.0]))
    array_layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(array_layer.W.tolist(), [[0.9, 1.9], [2.9, 3.9]])
    assert np.allclose(array_layer.b.tolist(), [4.9, 5.9])


def test_construction_rejects_a_size_smaller_than_two(layer_cls):

    with pytest.raises(AssertionError):
        layer_cls(1, 3)
