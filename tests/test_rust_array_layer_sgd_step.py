"""
RustArrayLayer.sgd_step: one fused call (pa.layer_sgd_step) in place of accumulate_gradient then
apply_accumulated_gradient at batch_size=1, which is what RustArrayNetworkBase.learn called before.
Checked exactly (tolist() ==, not approx), at the layer and at the network level, against that
unfused pair. Layers whose update isn't plain SGD keep the unfused pair; the last test makes sure
every such subclass says so explicitly.
"""

import importlib
import pkgutil
import random
import struct

import numpy as np
import pytest

import indrajala_math_rust as pa
import indrajala_ml.model
from indrajala_ml.model.adam_rust_array_layer import AdamRustArrayLayer
from indrajala_ml.model.adam_rust_array_multiclass_backprop_classifier_network import (
    AdamRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.array_layer import unfused_sgd_step
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
from indrajala_ml.model.l2_rust_array_layer import L2RustArrayLayer
from indrajala_ml.model.l2_rust_array_multiclass_backprop_classifier_network import (
    L2RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.momentum_rust_array_layer import MomentumRustArrayLayer
from indrajala_ml.model.momentum_rust_array_multiclass_backprop_classifier_network import (
    MomentumRustArrayMultiClassBackpropClassifierNetwork,
)
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

SIZE, INPUT_SIZE = 7, 11

LAYER_FACTORIES = {
    "plain": lambda: RustArrayLayer(SIZE, INPUT_SIZE),
    "relu": lambda: ReLURustArrayLayer(SIZE, INPUT_SIZE),
    "softmax": lambda: SoftmaxRustArrayLayer(SIZE, INPUT_SIZE),
    "cross-entropy": lambda: CrossEntropyRustArrayLayer(SIZE, INPUT_SIZE),
    "dropout": lambda: DropoutRustArrayLayer(SIZE, INPUT_SIZE, 0.5),
    "momentum": lambda: MomentumRustArrayLayer(SIZE, INPUT_SIZE, 0.9),
    "adam": lambda: AdamRustArrayLayer(SIZE, INPUT_SIZE, 0.9, 0.999, 1e-8),
    "l2": lambda: L2RustArrayLayer(SIZE, INPUT_SIZE, 0.01),
}

NETWORK_FACTORIES = {
    "plain": lambda: RustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3),
    "binary": lambda: RustArrayBackpropClassifierNetwork([5], 6),
    "relu": lambda: ReLURustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3),
    "softmax": lambda: SoftmaxRustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3),
    "cross-entropy": lambda: CrossEntropyRustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3),
    "cross-entropy binary": lambda: CrossEntropyRustArrayBackpropClassifierNetwork([5], 6),
    "momentum": lambda: MomentumRustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3, 0.9),
    "adam": lambda: AdamRustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3),
    "l2": lambda: L2RustArrayMultiClassBackpropClassifierNetwork([5, 4], 6, 3, 0.01),
    "conv": lambda: ConvRustArrayMultiClassBackpropClassifierNetwork(
        6, 6, [ConvSpec(3, 2), PoolSpec(2)], [4], 3
    ),
}


def _bits(nested):
    if isinstance(nested, list):
        return [_bits(item) for item in nested]
    return struct.pack("<d", nested)


def _random_layer_state(layer, rng: np.random.Generator):
    layer.W = pa.Array(rng.uniform(-1.0, 1.0, (SIZE, INPUT_SIZE)).tolist())
    layer.b = pa.Array(rng.uniform(-1.0, 1.0, SIZE).tolist())
    layer.delta = pa.Array(rng.uniform(-1.0, 1.0, SIZE).tolist())
    x = rng.uniform(0.0, 1.0, INPUT_SIZE)
    x[rng.random(INPUT_SIZE) < 0.3] = 0.0
    return pa.Array(x.tolist())


@pytest.mark.parametrize("name", LAYER_FACTORIES)
@pytest.mark.parametrize("seed", range(10))
def test_layer_sgd_step_is_bit_identical_to_accumulate_then_apply(name, seed):
    fused, unfused = LAYER_FACTORIES[name](), LAYER_FACTORIES[name]()
    x = _random_layer_state(fused, np.random.default_rng(seed))
    _random_layer_state(unfused, np.random.default_rng(seed))

    # two steps, so a stateful update (momentum's velocity, Adam's m/v/t) is exercised
    for learning_rate in (0.5, 0.1):
        fused.sgd_step(x, learning_rate)
        unfused_sgd_step(unfused, x, learning_rate)

    assert _bits(fused.W.tolist()) == _bits(unfused.W.tolist())
    assert _bits(fused.b.tolist()) == _bits(unfused.b.tolist())
    # accumulators stay fresh zeros either way, ready for the next step
    assert fused._grad_W.tolist() == unfused._grad_W.tolist() == [[0.0] * INPUT_SIZE] * SIZE
    assert fused._grad_b.tolist() == unfused._grad_b.tolist() == [0.0] * SIZE


def _sample(network, rng: random.Random):
    state = tuple(rng.uniform(0.0, 1.0) for _ in range(network.dimension))
    category = 1.0 if isinstance(network, RustArrayBackpropClassifierNetwork) else rng.randrange(3)
    return state, category


@pytest.mark.parametrize("name", NETWORK_FACTORIES)
def test_learn_is_bit_identical_to_the_unfused_step_after_every_step(name):
    # the reference network's layers run the pre-sgd_step loop: accumulate_gradient then
    # apply_accumulated_gradient(learning_rate, 1). Dropout networks are left out only because
    # their masks come from an unseeded RNG; DropoutRustArrayLayer is covered at the layer level.
    fused, unfused = NETWORK_FACTORIES[name](), NETWORK_FACTORIES[name]()
    fused.randomize()
    unfused.restore(fused.snapshot())
    for layer in _all_layers(unfused):
        layer.sgd_step = lambda x, learning_rate, layer=layer: unfused_sgd_step(layer, x, learning_rate)

    rng = random.Random(0)
    for _step in range(10):
        state, category = _sample(fused, rng)
        fused.learn(0.5, state, category)
        unfused.learn(0.5, state, category)
        for fused_layer, unfused_layer in zip(_all_layers(fused), _all_layers(unfused)):
            if hasattr(fused_layer, "W"):
                assert _bits(fused_layer.W.tolist()) == _bits(unfused_layer.W.tolist())
                assert _bits(fused_layer.b.tolist()) == _bits(unfused_layer.b.tolist())


def _all_layers(network):
    return getattr(network, "conv_layers", []) + list(network.layers)


def _all_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _all_subclasses(subclass)


def test_every_subclass_that_changes_the_update_overrides_sgd_step():
    # RustArrayLayer.sgd_step fuses plain-SGD accumulate + apply. A subclass that changes either
    # one and inherits that fused sgd_step would silently train with plain SGD instead.
    for module in pkgutil.iter_modules(indrajala_ml.model.__path__):
        importlib.import_module(f"indrajala_ml.model.{module.name}")
    subclasses = list(_all_subclasses(RustArrayLayer))
    assert {MomentumRustArrayLayer, AdamRustArrayLayer, L2RustArrayLayer} <= set(subclasses)

    for subclass in subclasses:
        changes_update = (
            subclass.accumulate_gradient is not RustArrayLayer.accumulate_gradient
            or subclass.apply_accumulated_gradient is not RustArrayLayer.apply_accumulated_gradient
        )
        if changes_update:
            assert subclass.sgd_step is not RustArrayLayer.sgd_step, subclass.__name__
