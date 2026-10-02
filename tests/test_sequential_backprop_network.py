"""
The pure-Python sequential networks (sequential_backprop_network.py): what each requires of its
output layer, that they save in format 2 only, and that every pure-Python preset is the sequential
network of its layer specs and update rule, by bits: the same layer classes, the same weights
after every learn and learn_batch step, and, for the fan-in-aware presets, the same randomize
draws. That makes the sequential network a faithful parity reference for the array networks
(tests/array_network_contract.py).
"""

import random
from collections.abc import Callable
from typing import Any

import pytest

from indrajala_ml.model.adam_backprop_classifier_network import AdamBackpropClassifierNetwork
from indrajala_ml.model.adam_conv_multiclass_backprop_classifier_network import (
    AdamConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.adam_multiclass_backprop_classifier_network import AdamMultiClassBackpropClassifierNetwork
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.binary_cross_entropy_backprop_classifier_network import (
    BinaryCrossEntropyBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.cross_entropy_conv_multiclass_backprop_classifier_network import (
    CrossEntropyConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_multiclass_backprop_classifier_network import (
    CrossEntropyMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.dropout_backprop_classifier_network import DropoutBackpropClassifierNetwork
from indrajala_ml.model.dropout_conv_multiclass_backprop_classifier_network import (
    DropoutConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.dropout_multiclass_backprop_classifier_network import (
    DropoutMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.fan_in_aware_backprop_classifier_network import FanInAwareBackpropClassifierNetwork
from indrajala_ml.model.l2_regularized_backprop_classifier_network import L2RegularizedBackpropClassifierNetwork
from indrajala_ml.model.l2_regularized_conv_multiclass_backprop_classifier_network import (
    L2RegularizedConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.l2_regularized_multiclass_backprop_classifier_network import (
    L2RegularizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_backprop_classifier_network import MomentumBackpropClassifierNetwork
from indrajala_ml.model.momentum_conv_multiclass_backprop_classifier_network import (
    MomentumConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_multiclass_backprop_classifier_network import (
    MomentumMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.relu_backprop_classifier_network import ReLUBackpropClassifierNetwork
from indrajala_ml.model.relu_conv_multiclass_backprop_classifier_network import (
    ReLUConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.relu_multiclass_backprop_classifier_network import ReLUMultiClassBackpropClassifierNetwork
from indrajala_ml.model.sequential_backprop_network import (
    SequentialBackpropClassifierNetwork,
    SequentialMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_conv_multiclass_backprop_classifier_network import (
    SoftmaxConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_multiclass_backprop_classifier_network import (
    SoftmaxMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Conv, Dense, LayerSpec, Pool
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import default_rng

DIMENSION = 6
CLASS_COUNT = 3
BOUNDS = [(-1.0, 1.0)] * DIMENSION
SIDE = 6
CONV_SPECS: list[Conv | Pool] = [Conv(2, 2), Pool(2, stride=1), Conv(2, 2)]

HIDDEN = [Dense(5), Dense(4)]
OUTPUT = Dense(CLASS_COUNT, output=True)
SINGLE = Dense(1, output=True)


# name -> (the preset, built; its equivalent: input shape, specs and rule; whether randomize is
# fan-in-aware, so the sequential network's own randomize must draw the same)
PRESETS: dict[str, tuple[Callable[[], Any], InputShape, list[LayerSpec], UpdateRule, bool]] = {
    "multiclass": (
        lambda: MultiClassBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, CLASS_COUNT),
        (DIMENSION,),
        [*HIDDEN, OUTPUT],
        SGD(),
        True,
    ),
    "softmax": (
        lambda: SoftmaxMultiClassBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, CLASS_COUNT),
        (DIMENSION,),
        [*HIDDEN, Dense(CLASS_COUNT, output=True, activation="softmax", loss="cross_entropy")],
        SGD(),
        True,
    ),
    "multiclass cross-entropy": (
        lambda: CrossEntropyMultiClassBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, CLASS_COUNT),
        (DIMENSION,),
        [*HIDDEN, Dense(CLASS_COUNT, output=True, loss="cross_entropy")],
        SGD(),
        True,
    ),
    "multiclass relu": (
        lambda: ReLUMultiClassBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, CLASS_COUNT),
        (DIMENSION,),
        [Dense(5, activation="relu"), Dense(4, activation="relu"), OUTPUT],
        SGD(),
        True,
    ),
    "multiclass dropout": (
        lambda: DropoutMultiClassBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, CLASS_COUNT, 0.3),
        (DIMENSION,),
        [Dense(5, dropout=0.3), Dense(4, dropout=0.3), OUTPUT],
        SGD(),
        True,
    ),
    "multiclass momentum": (
        lambda: MomentumMultiClassBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, CLASS_COUNT, 0.9),
        (DIMENSION,),
        [*HIDDEN, OUTPUT],
        Momentum(0.9),
        True,
    ),
    "multiclass adam": (
        lambda: AdamMultiClassBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, CLASS_COUNT, 0.8, 0.99, 1e-7),
        (DIMENSION,),
        [*HIDDEN, OUTPUT],
        Adam(0.8, 0.99, 1e-7),
        True,
    ),
    "multiclass l2": (
        lambda: L2RegularizedMultiClassBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, CLASS_COUNT, 0.02),
        (DIMENSION,),
        [*HIDDEN, OUTPUT],
        WeightDecay(0.02),
        True,
    ),
    "conv": (
        lambda: ConvMultiClassBackpropClassifierNetwork(SIDE, SIDE, CONV_SPECS, [4], CLASS_COUNT),
        (SIDE, SIDE, 1),
        [*CONV_SPECS, Dense(4), OUTPUT],
        SGD(),
        True,
    ),
    "momentum conv": (
        lambda: MomentumConvMultiClassBackpropClassifierNetwork(SIDE, SIDE, CONV_SPECS, [4], CLASS_COUNT, 0.9),
        (SIDE, SIDE, 1),
        [*CONV_SPECS, Dense(4), OUTPUT],
        Momentum(0.9),
        True,
    ),
    "adam conv": (
        lambda: AdamConvMultiClassBackpropClassifierNetwork(SIDE, SIDE, CONV_SPECS, [4], CLASS_COUNT, 0.8, 0.99, 1e-7),
        (SIDE, SIDE, 1),
        [*CONV_SPECS, Dense(4), OUTPUT],
        Adam(0.8, 0.99, 1e-7),
        True,
    ),
    "l2 conv": (
        lambda: L2RegularizedConvMultiClassBackpropClassifierNetwork(SIDE, SIDE, CONV_SPECS, [4], CLASS_COUNT, 0.02),
        (SIDE, SIDE, 1),
        [*CONV_SPECS, Dense(4), OUTPUT],
        WeightDecay(0.02),
        True,
    ),
    "relu conv": (
        lambda: ReLUConvMultiClassBackpropClassifierNetwork(SIDE, SIDE, CONV_SPECS, [4], CLASS_COUNT),
        (SIDE, SIDE, 1),
        [*CONV_SPECS, Dense(4, activation="relu"), OUTPUT],
        SGD(),
        True,
    ),
    "dropout conv": (
        lambda: DropoutConvMultiClassBackpropClassifierNetwork(SIDE, SIDE, CONV_SPECS, [4], CLASS_COUNT, 0.3),
        (SIDE, SIDE, 1),
        [*CONV_SPECS, Dense(4, dropout=0.3), OUTPUT],
        SGD(),
        True,
    ),
    "cross-entropy conv": (
        lambda: CrossEntropyConvMultiClassBackpropClassifierNetwork(SIDE, SIDE, CONV_SPECS, [4], CLASS_COUNT),
        (SIDE, SIDE, 1),
        [*CONV_SPECS, Dense(4), Dense(CLASS_COUNT, output=True, loss="cross_entropy")],
        SGD(),
        True,
    ),
    "softmax conv": (
        lambda: SoftmaxConvMultiClassBackpropClassifierNetwork(SIDE, SIDE, CONV_SPECS, [4], CLASS_COUNT),
        (SIDE, SIDE, 1),
        [*CONV_SPECS, Dense(4), Dense(CLASS_COUNT, output=True, activation="softmax", loss="cross_entropy")],
        SGD(),
        True,
    ),
    "single-output": (
        lambda: BackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS),
        (DIMENSION,),
        [*HIDDEN, SINGLE],
        SGD(),
        False,
    ),
    "fan-in-aware": (
        lambda: FanInAwareBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS),
        (DIMENSION,),
        [*HIDDEN, SINGLE],
        SGD(),
        True,
    ),
    "cross-entropy": (
        lambda: BinaryCrossEntropyBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS),
        (DIMENSION,),
        [*HIDDEN, Dense(1, output=True, loss="cross_entropy")],
        SGD(),
        False,
    ),
    "relu": (
        lambda: ReLUBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS),
        (DIMENSION,),
        [Dense(5, activation="relu"), Dense(4, activation="relu"), SINGLE],
        SGD(),
        False,
    ),
    "dropout": (
        lambda: DropoutBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, 0.3),
        (DIMENSION,),
        [Dense(5, dropout=0.3), Dense(4, dropout=0.3), SINGLE],
        SGD(),
        False,
    ),
    "momentum": (
        lambda: MomentumBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, 0.9),
        (DIMENSION,),
        [*HIDDEN, SINGLE],
        Momentum(0.9),
        False,
    ),
    "adam": (
        lambda: AdamBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, 0.8, 0.99, 1e-7),
        (DIMENSION,),
        [*HIDDEN, SINGLE],
        Adam(0.8, 0.99, 1e-7),
        False,
    ),
    "l2": (
        lambda: L2RegularizedBackpropClassifierNetwork([5, 4], DIMENSION, BOUNDS, 0.02),
        (DIMENSION,),
        [*HIDDEN, SINGLE],
        WeightDecay(0.02),
        False,
    ),
}


def _sequential(input_shape: InputShape, specs: list[LayerSpec], rule: UpdateRule) -> Any:
    output = specs[-1]
    assert isinstance(output, Dense)
    if output.size == 1:
        return SequentialBackpropClassifierNetwork(input_shape, specs, rule)
    return SequentialMultiClassBackpropClassifierNetwork(input_shape, specs, rule)


@pytest.mark.parametrize("name", PRESETS)
def test_every_pure_python_preset_is_the_sequential_network_of_its_specs_by_bits(name: str):
    build, input_shape, specs, rule, fan_in_aware = PRESETS[name]
    preset = build()
    sequential = _sequential(input_shape, specs, rule)

    assert [type(layer).__name__ for layer in sequential.trainable_layers] == [
        type(layer).__name__ for layer in preset.trainable_layers
    ]
    assert sequential.optimizer.rule == preset.optimizer.rule

    preset.rng = default_rng(0)
    preset.randomize()
    if fan_in_aware:
        sequential.rng = default_rng(0)
        sequential.randomize()
    else:
        sequential.restore(preset.snapshot())
    assert sequential.snapshot() == preset.snapshot()

    single_output = isinstance(preset, BackpropClassifierNetwork)
    rng = random.Random(4)

    def example() -> tuple[tuple[float, ...], Any]:
        state = tuple(rng.random() for _ in range(sequential.dimension))
        return state, float(rng.randrange(2)) if single_output else rng.randrange(CLASS_COUNT)

    # both generators are reseeded before each step, so dropout draws the same masks
    for step in range(6):
        state, target = example()
        for network in (preset, sequential):
            network.rng = default_rng(step)
            network.learn(0.3, state, target)
        assert sequential.snapshot() == preset.snapshot(), f"after learn step {step}"

    for step in range(4):
        batch = [example() for _ in range(5)]
        for network in (preset, sequential):
            network.rng = default_rng(100 + step)
            network.learn_batch(0.3, batch)
        assert sequential.snapshot() == preset.snapshot(), f"after learn_batch step {step}"

    state, _target = example()
    assert sequential.classify_state(state) == preset.classify_state(state)


def test_a_multiclass_network_needs_two_classes():
    with pytest.raises(AssertionError, match="class_count"):
        SequentialMultiClassBackpropClassifierNetwork((4,), [Dense(5), SINGLE], SGD())
    assert SequentialMultiClassBackpropClassifierNetwork((4,), [Dense(5), OUTPUT], SGD()).class_count == CLASS_COUNT


def test_a_single_output_network_needs_one_output_node():
    with pytest.raises(AssertionError, match="one-node"):
        SequentialBackpropClassifierNetwork((4,), [Dense(5), OUTPUT], SGD())


def test_the_specs_are_validated():
    with pytest.raises(AssertionError, match="dropout"):
        SequentialMultiClassBackpropClassifierNetwork((4,), [Dense(5, activation="relu", dropout=0.2), OUTPUT], SGD())


def test_multiclass_learn_drops_out_in_its_forward_pass_only():
    # as the single-output learn: a dropout layer trains with masks, and predicts without
    network = SequentialMultiClassBackpropClassifierNetwork((4,), [Dense(50, dropout=0.5), OUTPUT], SGD())
    network.rng = default_rng(0)
    network.randomize()
    hidden = network.hidden_layers[0]

    network.learn(0.1, (0.1, 0.2, 0.3, 0.4), 1)
    assert any(node.value() == 0.0 for node in hidden.nodes)

    network.predict_probabilities((0.1, 0.2, 0.3, 0.4))
    assert all(node.value() != 0.0 for node in hidden.nodes)
