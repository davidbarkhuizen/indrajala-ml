"""
The array optimizers (optimizers.py) under each stateful or decaying rule, Momentum, Adam and
WeightDecay: stepped alike, each matches the pure-Python optimizer stepping a BackpropLayer at every
step and every batch, and each resets the accumulated gradient. Then each rule's own property.
"""

import random

import numpy as np
import pytest

from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.state_layer import StateLayer
from indrajala_ml.model.update_rules import Adam, Momentum, UpdateRule, WeightDecay
from tests.helpers import Backend, LayerOptimizer, array_layer_like

MOMENTUM = Momentum(0.5)
ADAM = Adam(0.9, 0.999, 1e-8)
WEIGHT_DECAY = WeightDecay(0.05)

# each rule with its seeds for the every-step and the every-batch comparisons
RULES = [
    pytest.param(MOMENTUM, 31, 32, id="momentum"),
    pytest.param(ADAM, 11, 12, id="adam"),
    pytest.param(WEIGHT_DECAY, 21, 22, id="weight_decay"),
]

# the layers the optimizer steps, as the pure-Python optimizer steps a BackpropLayer
LayerCls = type[ArrayLayer] | type[RustArrayLayer]
LAYER_CLS: dict[str, LayerCls] = {"numpy": ArrayLayer, "rust": RustArrayLayer}


def _random_layer_pair(
    rng: random.Random, dimension: int, size: int, backend: Backend
) -> tuple[StateLayer, BackpropLayer, ArrayLayer | RustArrayLayer]:
    state_layer = StateLayer(dimension, [(-10.0, 10.0)] * dimension)
    backprop_layer = BackpropLayer(size=size, input_layer=state_layer)
    for node in backprop_layer.nodes:
        node.update_input_weights([rng.uniform(-3.0, 3.0) for _ in range(dimension)])
        node.bias = rng.uniform(-3.0, 3.0)
    return state_layer, backprop_layer, array_layer_like(LAYER_CLS[backend.name], backprop_layer, backend)


def _assert_weights_match(array_layer: ArrayLayer | RustArrayLayer, backprop_layer: BackpropLayer) -> None:
    expected_W = np.array([node.input_node_weights for node in backprop_layer.nodes])
    expected_b = np.array([node.bias for node in backprop_layer.nodes])
    assert np.allclose(array_layer.W.tolist(), expected_W, rtol=1e-9, atol=1e-12)
    assert np.allclose(array_layer.b.tolist(), expected_b, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("rule, step_seed, _batch_seed", RULES)
def test_accumulate_then_apply_at_batch_size_one_matches_the_pure_python_optimizer_at_every_step(
    rule: UpdateRule, step_seed: int, _batch_seed: int, backend: Backend
):

    # compared after every step: a rule's state (velocity, m/v/t), or a penalty on the current W,
    # only shows a mistake across repeated steps
    rng = random.Random(step_seed)
    dimension = 5
    size = 4

    state_layer, backprop_layer, array_layer = _random_layer_pair(rng, dimension, size, backend)
    optimizer = LayerOptimizer(array_layer, rule)
    node_optimizer = LayerOptimizer(backprop_layer, rule)
    learning_rate = rng.uniform(0.001, 1.0)

    for _ in range(10):
        x = [rng.uniform(-10.0, 10.0) for _ in range(dimension)]
        state_layer.update_state(tuple(x))
        deltas = [rng.uniform(-5.0, 5.0) for _ in range(size)]

        for node, delta in zip(backprop_layer.nodes, deltas):
            node.delta = delta
            node.accumulate_gradient()
        node_optimizer.apply(learning_rate, batch_size=1)

        array_layer.delta = backend.owned(deltas)
        array_layer.accumulate_gradient(backend.owned(x))
        optimizer.apply(learning_rate, batch_size=1)

        _assert_weights_match(array_layer, backprop_layer)


@pytest.mark.parametrize("rule, _step_seed, batch_seed", RULES)
def test_accumulate_across_a_batch_then_apply_matches_the_pure_python_optimizer_at_every_batch(
    rule: UpdateRule, _step_seed: int, batch_seed: int, backend: Backend
):

    rng = random.Random(batch_seed)
    dimension = 4
    size = 3
    batch_size = 6

    state_layer, backprop_layer, array_layer = _random_layer_pair(rng, dimension, size, backend)
    optimizer = LayerOptimizer(array_layer, rule)
    node_optimizer = LayerOptimizer(backprop_layer, rule)
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

        node_optimizer.apply(learning_rate, batch_size)
        optimizer.apply(learning_rate, batch_size)

        _assert_weights_match(array_layer, backprop_layer)


@pytest.mark.parametrize("rule", [MOMENTUM, ADAM, WEIGHT_DECAY], ids=["momentum", "adam", "weight_decay"])
def test_apply_accumulated_gradient_resets_the_accumulator(rule: UpdateRule, layer_cls: LayerCls, backend: Backend):

    array_layer = layer_cls(3, 2)
    array_layer.delta = backend.owned([0.1, 0.2, 0.3])
    array_layer.accumulate_gradient(backend.owned([1.0, 2.0]))
    LayerOptimizer(array_layer, rule).apply(0.1, batch_size=1)

    assert np.allclose(array_layer.grad_W.tolist(), np.zeros((3, 2)))
    assert np.allclose(array_layer.grad_b.tolist(), np.zeros(3))


def test_momentum_velocity_starts_at_zero_so_the_first_step_is_plain_sgd(layer_cls: LayerCls, backend: Backend):

    array_layer = layer_cls(2, 2)
    array_layer.W = backend.owned([[1.0, 2.0], [3.0, 4.0]])
    array_layer.b = backend.owned([5.0, 6.0])
    array_layer.grad_W = backend.owned([[1.0, 1.0], [1.0, 1.0]])
    array_layer.grad_b = backend.owned([1.0, 1.0])

    LayerOptimizer(array_layer, Momentum(0.9)).apply(learning_rate=0.1, batch_size=1)

    assert np.allclose(array_layer.W.tolist(), [[0.9, 1.9], [2.9, 3.9]])
    assert np.allclose(array_layer.b.tolist(), [4.9, 5.9])


def test_adam_step_count_increments_once_per_update(layer_cls: LayerCls, backend: Backend):

    array_layer = layer_cls(3, 2)
    optimizer = LayerOptimizer(array_layer, ADAM)
    assert optimizer.optimizer.t == 0

    array_layer.delta = backend.owned([0.1, 0.2, 0.3])
    array_layer.accumulate_gradient(backend.owned([1.0, 2.0]))
    optimizer.apply(0.1, batch_size=1)
    assert optimizer.optimizer.t == 1

    array_layer.delta = backend.owned([0.1, 0.2, 0.3])
    array_layer.accumulate_gradient(backend.owned([1.0, 2.0]))
    optimizer.apply(0.1, batch_size=1)
    assert optimizer.optimizer.t == 2


def test_weight_decay_never_regularizes_the_bias(layer_cls: LayerCls, backend: Backend):

    # with a zero weight gradient and a nonzero bias gradient, the bias moves by plain SGD: the
    # penalty applies to W only
    array_layer = layer_cls(2, 2)
    array_layer.W = backend.owned([[1.0, 2.0], [3.0, 4.0]])
    array_layer.b = backend.owned([5.0, 6.0])
    array_layer.delta = backend.owned([0.0, 0.0])
    array_layer.accumulate_gradient(backend.owned([0.0, 0.0]))
    array_layer.grad_b = backend.owned([2.0, 4.0])

    LayerOptimizer(array_layer, WeightDecay(0.5)).apply(learning_rate=0.1, batch_size=1)

    assert np.allclose(array_layer.b.tolist(), [5.0 - 0.1 * 2.0, 6.0 - 0.1 * 4.0])
