import json
import random
from pathlib import Path

from indrajala_ml.model.classifier_protocols import State
from indrajala_ml.model.conv_layer import ConvLayer, ConvSpec
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import (
    ConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_layer import MaxPoolLayer, PoolSpec
from indrajala_ml.model.momentum_conv_multiclass_backprop_classifier_network import (
    MomentumConvMultiClassBackpropClassifierNetwork,
)
from tests.helpers import assert_save_and_load_round_trip

# pooling (overlapping), stride, a multi-channel second conv layer and two dense layers
CONV_SPECS = [ConvSpec(3, 3), PoolSpec(2, stride=1), ConvSpec(2, 4, stride=2)]
DENSE_LAYER_SIZES = [8, 6]
CLASS_COUNT = 10


def _matching_networks(
    momentum: float,
) -> tuple[ConvMultiClassBackpropClassifierNetwork, MomentumConvMultiClassBackpropClassifierNetwork]:
    random.seed(0)
    plain = ConvMultiClassBackpropClassifierNetwork.randomized(8, 8, CONV_SPECS, DENSE_LAYER_SIZES, CLASS_COUNT)
    momentum_network = MomentumConvMultiClassBackpropClassifierNetwork(
        8, 8, CONV_SPECS, DENSE_LAYER_SIZES, CLASS_COUNT, momentum=momentum
    )
    momentum_network.restore(plain.snapshot())
    return plain, momentum_network


def _rows(count: int) -> list[tuple[State, int]]:
    rng = random.Random(1)
    return [(tuple(rng.uniform(0.0, 1.0) for _ in range(64)), rng.randrange(CLASS_COUNT)) for _ in range(count)]


def test_the_hooks_build_momentum_kernels_and_dense_layers_and_leave_pooling_alone():

    network = MomentumConvMultiClassBackpropClassifierNetwork(
        8, 8, CONV_SPECS, DENSE_LAYER_SIZES, CLASS_COUNT, momentum=0.9
    )
    first, pool, last = network.conv_layers

    for layer in (first, last):
        assert isinstance(layer, ConvLayer)
        assert all(type(kernel).__name__ == "MomentumConvKernel" for kernel in layer.kernels)
    assert type(pool) is MaxPoolLayer
    for layer in [*network.hidden_layers[3:], network.output_layer]:
        assert all(type(node).__name__ == "MomentumBackpropNode" for node in layer.nodes)


def test_zero_momentum_is_bit_identical_to_the_plain_conv_network_through_learn():

    plain, momentum_network = _matching_networks(momentum=0.0)

    for state, label in _rows(8):
        plain.learn(0.1, state, label)
        momentum_network.learn(0.1, state, label)
        assert momentum_network.snapshot() == plain.snapshot()


def test_zero_momentum_is_bit_identical_to_the_plain_conv_network_through_learn_batch():

    # batches of 6, not a power of two, so g / B rounds; at momentum 0.0 eq. (9) is exactly
    # SGD's w - lr * (g / B)
    plain, momentum_network = _matching_networks(momentum=0.0)
    rows = _rows(24)

    for start in range(0, len(rows), 6):
        batch = rows[start : start + 6]
        plain.learn_batch(0.1, batch)
        momentum_network.learn_batch(0.1, batch)
        assert momentum_network.snapshot() == plain.snapshot()


def test_momentum_first_matches_plain_sgd_and_then_moves_every_trainable_layer_off_it():

    # the velocity starts at zero, so step 1 is SGD bit for bit; from step 2 the carried velocity
    # changes every conv kernel and every dense layer, so none still uses the plain update
    plain, momentum_network = _matching_networks(momentum=0.9)
    rows = _rows(12)

    plain.learn_batch(0.1, rows[:6])
    momentum_network.learn_batch(0.1, rows[:6])
    assert momentum_network.snapshot() == plain.snapshot()

    plain.learn_batch(0.1, rows[6:])
    momentum_network.learn_batch(0.1, rows[6:])
    for index, (momentum_entry, plain_entry) in enumerate(zip(momentum_network.snapshot(), plain.snapshot())):
        if isinstance(momentum_network.trainable_layers[index], MaxPoolLayer):
            continue
        assert momentum_entry != plain_entry, f"trainable layer {index} took the plain update"


def test_save_and_load_round_trip_keeps_momentum(tmp_path: Path):

    random.seed(0)
    network = MomentumConvMultiClassBackpropClassifierNetwork.randomized(
        8, 8, CONV_SPECS, DENSE_LAYER_SIZES, CLASS_COUNT, momentum=0.9
    )
    states = [state for state, _label in _rows(3)]

    loaded = assert_save_and_load_round_trip(
        network, MomentumConvMultiClassBackpropClassifierNetwork.load, tmp_path, "momentum_conv.json", states
    )

    assert loaded.momentum == 0.9
    assert loaded.conv_specs == network.conv_specs
    assert json.loads((tmp_path / "momentum_conv.json").read_text())["momentum"] == 0.9
    # the loaded layers train with the saved momentum: two batches match a fresh 0.9 network's
    reference = MomentumConvMultiClassBackpropClassifierNetwork(
        8, 8, CONV_SPECS, DENSE_LAYER_SIZES, CLASS_COUNT, momentum=0.9
    )
    reference.restore(loaded.snapshot())
    rows = _rows(6)
    for _ in range(2):
        loaded.learn_batch(0.1, rows)
        reference.learn_batch(0.1, rows)
    assert loaded.snapshot() == reference.snapshot()
