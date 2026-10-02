import random

import numpy as np

from indrajala_ml.model.layers.numpy.array_layer import ArrayLayer
from indrajala_ml.model.layers.numpy.cross_entropy_array_layer import CrossEntropyArrayLayer
from indrajala_ml.model.layers.python.cross_entropy_output_layer import CrossEntropyOutputLayer
from indrajala_ml.model.layers.python.state_layer import StateLayer
from indrajala_ml.model.layers.rust.cross_entropy_rust_array_layer import CrossEntropyRustArrayLayer
from indrajala_ml.model.layers.rust.rust_array_layer import RustArrayLayer
from tests.helpers import Backend, LayerOptimizer, array_layer_like

LayerCls = type[CrossEntropyArrayLayer] | type[CrossEntropyRustArrayLayer]
LAYER_CLS: dict[str, LayerCls] = {"numpy": CrossEntropyArrayLayer, "rust": CrossEntropyRustArrayLayer}
BASE_LAYER_CLS = {"numpy": ArrayLayer, "rust": RustArrayLayer}


def test_compute_output_delta_matches_cross_entropy_output_node_across_a_random_sweep(backend: Backend):

    rng = random.Random(60)
    dimension = 4
    size = 5

    for _ in range(100):
        state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
        cross_entropy_layer = CrossEntropyOutputLayer(size, state_layer)

        for node in cross_entropy_layer.nodes:
            node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
            node.bias = rng.uniform(-3.0, 3.0)

        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        cross_entropy_layer.forward()

        targets = [rng.choice([0.0, 1.0]) for _ in range(size)]
        expected: list[float] = []
        for node, target in zip(cross_entropy_layer.nodes, targets):
            node.compute_output_delta(target)
            expected.append(node.delta)

        array_layer = array_layer_like(LAYER_CLS[backend.name], cross_entropy_layer, backend)
        array_layer.forward(backend.owned(x))
        array_layer.compute_output_delta(backend.owned(targets))

        assert np.allclose(array_layer.delta.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_compute_output_delta_batch_matches_per_row_single_example_results_stacked(
    layer_cls: LayerCls, backend: Backend
):

    rng = random.Random(61)
    dimension = 4
    size = 3
    batch_size = 5

    array_layer = layer_cls(size, dimension)
    array_layer.W = backend.owned([[rng.uniform(-3.0, 3.0) for _ in range(dimension)] for _ in range(size)])
    array_layer.b = backend.owned([rng.uniform(-3.0, 3.0) for _ in range(size)])

    X = [[rng.uniform(-10.0, 10.0) for _ in range(dimension)] for _ in range(batch_size)]
    reference_rows = [[rng.choice([0.0, 1.0]) for _ in range(size)] for _ in range(batch_size)]

    expected: list[list[float]] = []
    for row in range(batch_size):
        array_layer.forward(backend.owned(X[row]))
        array_layer.compute_output_delta(backend.owned(reference_rows[row]))
        expected.append(array_layer.delta.tolist())

    array_layer.forward_batch(backend.owned(X))
    array_layer.compute_output_delta_batch(backend.owned(reference_rows))
    assert np.allclose(array_layer.delta_batch.tolist(), expected, rtol=1e-9, atol=1e-12)


def test_forward_is_inherited_unchanged_from_array_layer(layer_cls: LayerCls, backend: Backend):

    base = BASE_LAYER_CLS[backend.name]
    assert layer_cls.__mro__[1] is base
    assert layer_cls.forward is base.forward
    assert layer_cls.forward_batch is base.forward_batch


def test_compute_hidden_delta_is_inherited_unchanged_from_array_layer(layer_cls: LayerCls, backend: Backend):

    base = BASE_LAYER_CLS[backend.name]
    assert layer_cls.compute_hidden_delta is base.compute_hidden_delta
    assert layer_cls.compute_hidden_delta_batch is base.compute_hidden_delta_batch


def test_the_weight_update_is_array_layers(layer_cls: LayerCls, backend: Backend):

    array_layer = layer_cls(2, 2)
    array_layer.W = backend.owned([[1.0, 2.0], [3.0, 4.0]])
    array_layer.b = backend.owned([5.0, 6.0])
    array_layer.delta = backend.owned([1.0, 1.0])
    array_layer.accumulate_gradient(backend.owned([1.0, 1.0]))
    LayerOptimizer(array_layer).apply(learning_rate=0.1, batch_size=1)

    assert np.allclose(array_layer.W.tolist(), [[0.9, 1.9], [2.9, 3.9]])
    assert np.allclose(array_layer.b.tolist(), [4.9, 5.9])
