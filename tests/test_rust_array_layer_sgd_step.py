"""
RustOptimizer.step_single: under SGD, one fused call (pa.layer_sgd_step) in place of
accumulate_gradient then the optimizer's apply at batch_size=1, which is what a Rust network's
learn() called before the step was fused. Checked exactly (bit for bit, not approx), at the layer
and at the network level, against that unfused pair, for every rule. The fused call is used iff
the rule is SGD and the layer is dense: the other rules, and the conv layers (whose gradient sums
over output positions), keep the unfused pair.
"""

import importlib
import pkgutil
import random
import struct
from collections.abc import Callable
from typing import Any, cast

import indrajala_math_rust as pa
import numpy as np
import pytest

import indrajala_ml.model
from indrajala_ml.model.adam_rust_array_multiclass_backprop_classifier_network import (
    AdamRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.array_backend import RUST
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_rust_array_backprop_classifier_network import (
    CrossEntropyRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_rust_array_layer import CrossEntropyRustArrayLayer
from indrajala_ml.model.cross_entropy_rust_array_multiclass_backprop_classifier_network import (
    CrossEntropyRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.l2_rust_array_multiclass_backprop_classifier_network import (
    L2RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.momentum_conv_rust_array_multiclass_backprop_classifier_network import (
    MomentumConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_rust_array_multiclass_backprop_classifier_network import (
    MomentumRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.optimizers import RustOptimizer
from indrajala_ml.model.relu_rust_array_layer import ReLURustArrayLayer
from indrajala_ml.model.relu_rust_array_multiclass_backprop_classifier_network import (
    ReLURustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_rust_array_layer import SoftmaxRustArrayLayer
from indrajala_ml.model.softmax_rust_array_multiclass_backprop_classifier_network import (
    SoftmaxRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from tests.helpers import all_subclasses

SIZE, INPUT_SIZE = 7, 11

LAYER_FACTORIES = {
    "plain": lambda: RustArrayLayer(SIZE, INPUT_SIZE),
    "relu": lambda: ReLURustArrayLayer(SIZE, INPUT_SIZE),
    "softmax": lambda: SoftmaxRustArrayLayer(SIZE, INPUT_SIZE),
    "cross-entropy": lambda: CrossEntropyRustArrayLayer(SIZE, INPUT_SIZE),
    "dropout": lambda: DropoutRustArrayLayer(SIZE, INPUT_SIZE, 0.5),
}

RULES: dict[str, UpdateRule] = {
    "sgd": SGD(),
    "momentum": Momentum(0.9),
    "adam": Adam(0.9, 0.999, 1e-8),
    "weight decay": WeightDecay(0.01),
}

# (network factory, its rule's name)
NETWORK_FACTORIES: dict[str, tuple[Callable[[], Any], str]] = {
    "plain": (lambda: RustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3), "sgd"),
    "binary": (lambda: RustArrayBackpropClassifierNetwork([5], 6), "sgd"),
    "relu": (lambda: ReLURustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3), "sgd"),
    "softmax": (lambda: SoftmaxRustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3), "sgd"),
    "cross-entropy": (lambda: CrossEntropyRustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3), "sgd"),
    "cross-entropy binary": (lambda: CrossEntropyRustArrayBackpropClassifierNetwork([5], 6), "sgd"),
    "momentum": (lambda: MomentumRustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3, 0.9), "momentum"),
    "adam": (lambda: AdamRustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3), "adam"),
    "l2": (lambda: L2RustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3, 0.01), "weight decay"),
    "conv": (
        lambda: ConvRustArrayMultiClassBackpropClassifierNetwork(6, 6, [ConvSpec(3, 2), PoolSpec(2)], [4], 3),
        "sgd",
    ),
    "momentum conv": (
        lambda: MomentumConvRustArrayMultiClassBackpropClassifierNetwork(
            6, 6, [ConvSpec(3, 2), PoolSpec(2)], [4], 3, 0.9
        ),
        "momentum",
    ),
}


def _bits(nested: Any) -> Any:
    if isinstance(nested, list):
        return [_bits(item) for item in cast("list[Any]", nested)]
    return struct.pack("<d", nested)


def _random_layer_state(layer: RustArrayLayer, rng: np.random.Generator) -> pa.Array:
    layer.W = pa.Array(rng.uniform(-1.0, 1.0, (SIZE, INPUT_SIZE)).tolist())
    layer.b = pa.Array(rng.uniform(-1.0, 1.0, SIZE).tolist())
    layer.delta = pa.Array(rng.uniform(-1.0, 1.0, SIZE).tolist())
    x = rng.uniform(0.0, 1.0, INPUT_SIZE)
    x[rng.random(INPUT_SIZE) < 0.3] = 0.0
    return pa.Array(x.tolist())


def _unfused_step_single(optimizer: RustOptimizer) -> Callable[[int, Any, pa.Array, float], None]:
    # the step before it was fused: accumulate, then apply at batch_size=1
    def step_single(index: int, layer: Any, input_activation: pa.Array, learning_rate: float) -> None:
        layer.accumulate_gradient(input_activation)
        optimizer.apply(index, layer, learning_rate, 1)

    return step_single


@pytest.mark.parametrize("rule_name", RULES)
@pytest.mark.parametrize("name", LAYER_FACTORIES)
@pytest.mark.parametrize("seed", range(10))
def test_step_single_is_bit_identical_to_accumulate_then_apply(name: str, rule_name: str, seed: int):
    fused, unfused = LAYER_FACTORIES[name](), LAYER_FACTORIES[name]()
    x = _random_layer_state(fused, np.random.default_rng(seed))
    _random_layer_state(unfused, np.random.default_rng(seed))
    fused_optimizer, unfused_optimizer = RUST.optimizer(RULES[rule_name]), RUST.optimizer(RULES[rule_name])
    unfused_step_single = _unfused_step_single(unfused_optimizer)

    # two steps, so a stateful rule (momentum's velocity, Adam's m/v/t) is exercised
    for learning_rate in (0.5, 0.1):
        fused_optimizer.begin_step()
        fused_optimizer.step_single(0, fused, x, learning_rate)
        unfused_optimizer.begin_step()
        unfused_step_single(0, unfused, x, learning_rate)

    assert _bits(fused.W.tolist()) == _bits(unfused.W.tolist())
    assert _bits(fused.b.tolist()) == _bits(unfused.b.tolist())
    # accumulators stay fresh zeros either way, ready for the next step
    assert fused.grad_W.tolist() == unfused.grad_W.tolist() == [[0.0] * INPUT_SIZE] * SIZE
    assert fused.grad_b.tolist() == unfused.grad_b.tolist() == [0.0] * SIZE


# a network of any of NETWORK_FACTORIES' classes
def _sample(network: Any, rng: random.Random) -> tuple[tuple[float, ...], float | int]:
    state = tuple(rng.uniform(0.0, 1.0) for _ in range(network.dimension))
    category = 1.0 if isinstance(network, RustArrayBackpropClassifierNetwork) else rng.randrange(3)
    return state, category


@pytest.mark.parametrize("name", NETWORK_FACTORIES)
def test_learn_is_bit_identical_to_the_unfused_step_after_every_step(name: str):
    # the reference network's optimizer runs the unfused step. Dropout networks are left out only
    # because their masks come from an unseeded RNG; DropoutRustArrayLayer is covered at the layer
    # level.
    factory, _rule_name = NETWORK_FACTORIES[name]
    fused, unfused = factory(), factory()
    fused.randomize()
    unfused.restore(fused.snapshot())
    unfused.optimizer.step_single = _unfused_step_single(unfused.optimizer)

    rng = random.Random(0)
    for _step in range(10):
        state, category = _sample(fused, rng)
        fused.learn(0.5, state, category)
        unfused.learn(0.5, state, category)
        for fused_layer, unfused_layer in zip(fused.layers, unfused.layers):
            if hasattr(fused_layer, "W"):
                assert _bits(fused_layer.W.tolist()) == _bits(unfused_layer.W.tolist())
                assert _bits(fused_layer.b.tolist()) == _bits(unfused_layer.b.tolist())


@pytest.mark.parametrize("name", NETWORK_FACTORIES)
def test_the_fused_step_is_used_iff_the_rule_is_sgd(name: str, monkeypatch: pytest.MonkeyPatch):
    # one fused call per dense layer per learn() under SGD, none under any other rule; conv layers
    # never take it
    factory, rule_name = NETWORK_FACTORIES[name]
    network = factory()
    network.randomize()
    assert network.optimizer.rule == RULES[rule_name]

    calls: list[None] = []
    layer_sgd_step = pa.layer_sgd_step

    def counting_layer_sgd_step(*args: Any) -> Any:
        calls.append(None)
        return layer_sgd_step(*args)

    monkeypatch.setattr(pa, "layer_sgd_step", counting_layer_sgd_step)
    network.learn(0.5, *_sample(network, random.Random(0)))

    dense_layers = [layer for layer in network.layers if isinstance(layer, RustArrayLayer)]
    assert len(calls) == (len(dense_layers) if rule_name == "sgd" else 0)


def test_every_network_has_its_backends_optimizer():
    for factory, rule_name in NETWORK_FACTORIES.values():
        network = factory()
        assert type(network.optimizer) is type(RUST.optimizer(SGD())) and network.optimizer.rule == RULES[rule_name]


def test_no_dense_rust_layer_changes_accumulate_gradient():
    # the fused step computes RustArrayLayer.accumulate_gradient's plain outer product. A dense
    # subclass that changed it would train silently wrong under SGD, unless RustOptimizer.step_single
    # stopped fusing for it.
    for module in pkgutil.iter_modules(indrajala_ml.model.__path__):
        importlib.import_module(f"indrajala_ml.model.{module.name}")
    subclasses = list(all_subclasses(RustArrayLayer))
    assert {ReLURustArrayLayer, SoftmaxRustArrayLayer, DropoutRustArrayLayer} <= set(subclasses)

    for subclass in subclasses:
        assert subclass.accumulate_gradient is RustArrayLayer.accumulate_gradient, subclass.__name__
