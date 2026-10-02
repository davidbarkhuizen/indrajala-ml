"""
SequentialArrayNetwork (sequential_array_network.py): the class it builds for each shape and
backend, what each shape requires of the output layer, and a combination no preset has, trained
and matched against its pure-Python reference (sequential_backprop_network.py). That each preset
equals its sequential network by bits is in the preset's own tests (array_network_contract.py,
and the conv tests).
"""

import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.digits_data import load_digits_dataset
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.sequential_array_network import (
    SequentialArrayBackpropClassifierNetwork,
    SequentialArrayNetwork,
    SequentialRustArrayBackpropClassifierNetwork,
    SequentialRustArrayMultiClassBackpropClassifierNetwork,
    SequentialVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Conv, Dense, LayerSpec, Pool
from indrajala_ml.model.specs.update_rules import SGD, Adam
from tests.helpers import Backend, assert_conv_array_network_weights_match, conv_reference
from tests.test_conv_array_multiclass_backprop_model import PROBABILITY_ATOL, WEIGHT_ATOL

MULTICLASS = [Dense(5), Dense(3, output=True)]
SINGLE_OUTPUT = [Dense(5), Dense(1, output=True)]


@pytest.mark.parametrize(
    "shape, backend, cls",
    [
        ("multiclass", NUMPY, SequentialVectorizedMultiClassBackpropClassifierNetwork),
        ("multiclass", RUST, SequentialRustArrayMultiClassBackpropClassifierNetwork),
        ("single_output", NUMPY, SequentialArrayBackpropClassifierNetwork),
        ("single_output", RUST, SequentialRustArrayBackpropClassifierNetwork),
    ],
)
def test_it_builds_the_class_for_its_shape_and_backend(shape: Any, backend: Backend, cls: type[Any]):
    layers = MULTICLASS if shape == "multiclass" else SINGLE_OUTPUT
    network = SequentialArrayNetwork((4,), layers, SGD(), shape=shape, backend=backend)

    assert type(network) is cls
    assert network.layer_specs == layers and network.optimizer.rule == SGD()


def test_it_rejects_an_unknown_shape():
    with pytest.raises(AssertionError, match="shape"):
        SequentialArrayNetwork((4,), MULTICLASS, SGD(), shape="multi")  # pyright: ignore[reportArgumentType]


def test_a_multiclass_network_needs_two_classes(backend: Backend):
    with pytest.raises(AssertionError, match="class_count"):
        SequentialArrayNetwork((4,), SINGLE_OUTPUT, SGD(), backend=backend)
    assert SequentialArrayNetwork((4,), MULTICLASS, SGD(), backend=backend).class_count == 3


def test_a_single_output_network_needs_one_output_node(backend: Backend):
    with pytest.raises(AssertionError, match="one-node"):
        SequentialArrayNetwork((4,), MULTICLASS, SGD(), shape="single_output", backend=backend)


def test_the_specs_are_validated(backend: Backend):
    with pytest.raises(AssertionError, match="dropout"):
        SequentialArrayNetwork(
            (4,), [Dense(5, activation="relu", dropout=0.2), *MULTICLASS[1:]], SGD(), backend=backend
        )


def _no_preset_layers(drop_probability: float) -> list[LayerSpec]:
    # conv, pool, a ReLU layer, a dropout layer and a softmax output: a combination no preset has
    return [
        Conv(3, 2),
        Pool(2),
        Dense(6, activation="relu"),
        Dense(5, dropout=drop_probability),
        Dense(3, output=True, activation="softmax", loss="cross_entropy"),
    ]


def _digits_rows() -> list[tuple[tuple[float, ...], int]]:
    # real UCI digits, as the conv parity tests use, with three classes
    return [(state, label % 3) for state, label in load_digits_dataset()[:120]]


def test_a_combination_no_preset_has_trains(backend: Backend):
    # the combination under Adam: every weight moves, and the network still classifies
    layers = _no_preset_layers(0.2)
    network = SequentialArrayNetwork((8, 8, 1), layers, Adam(), backend=backend)
    network.rng = backend.default_rng(0)
    network.randomize()
    before = [[np.asarray(array.tolist()) for array in entry] for entry in network.snapshot()]

    rng = random.Random(0)
    examples = [(tuple(rng.random() for _ in range(64)), rng.randrange(3)) for _ in range(16)]
    for state, category in examples[:8]:
        network.learn(0.01, state, category)
    network.learn_batch(0.01, examples[8:])

    after = [[np.asarray(array.tolist()) for array in entry] for entry in network.snapshot()]
    assert [len(entry) for entry in after] == [2, 0, 2, 2, 2]
    for entry_before, entry_after in zip(before, after):
        for array_before, array_after in zip(entry_before, entry_after):
            assert not np.array_equal(array_before, array_after)
    probabilities = network.predict_probabilities(examples[0][0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert network.classify_state(examples[0][0]) in range(3)


def test_a_combination_no_preset_has_predicts_as_its_pure_python_reference(backend: Backend):
    # at eval, where dropout does nothing: the pure-Python dropout layers draw their masks from
    # Python's random, so training with dropout can't be compared (the next test trains at 0.0)
    layers = _no_preset_layers(0.2)
    network = SequentialArrayNetwork((8, 8, 1), layers, Adam(), backend=backend)
    reference = conv_reference(random.Random(0), network, (8, 8, 1), layers, Adam(), backend.owned)

    for state, _label in _digits_rows()[:50]:
        np.testing.assert_allclose(
            network.predict_probabilities(state), reference.predict_probabilities(state), rtol=0, atol=PROBABILITY_ATOL
        )
        assert network.classify_state(state) == reference.classify_state(state)


def test_a_combination_no_preset_has_trains_as_its_pure_python_reference(backend: Backend):
    # every step of learn and learn_batch under Adam, with a dropout layer that keeps every node
    # (drop_probability 0.0), so both implementations' masks agree. Over these runs both backends
    # stay within 2.2e-16 of the reference in probabilities (the test above) and 1.6e-15 in weights
    layers = _no_preset_layers(0.0)
    network = SequentialArrayNetwork((8, 8, 1), layers, Adam(), backend=backend)
    reference = conv_reference(random.Random(1), network, (8, 8, 1), layers, Adam(), backend.owned)
    rows = _digits_rows()

    for state, label in rows[:40]:
        reference.learn(0.01, state, label)
        network.learn(0.01, state, label)
        assert_conv_array_network_weights_match(reference, network, rtol=0, atol=WEIGHT_ATOL)

    for start in range(40, len(rows), 8):
        reference.learn_batch(0.01, rows[start : start + 8])
        network.learn_batch(0.01, rows[start : start + 8])
        assert_conv_array_network_weights_match(reference, network, rtol=0, atol=WEIGHT_ATOL)
