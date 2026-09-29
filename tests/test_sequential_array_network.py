"""
SequentialArrayNetwork (sequential_array_network.py): the class it builds for each shape and
backend, what each shape requires of the output layer, and a combination no preset has, trained.
That each preset equals its sequential network by bits is in the preset's own tests
(array_network_contract.py, and the conv tests).
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.layer_specs import Conv, Dense, Pool
from indrajala_ml.model.sequential_array_network import (
    SequentialArrayBackpropClassifierNetwork,
    SequentialArrayNetwork,
    SequentialRustArrayBackpropClassifierNetwork,
    SequentialRustArrayMultiClassBackpropClassifierNetwork,
    SequentialVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import SGD, Adam
from tests.helpers import Backend

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


def test_it_saves_in_format_2_only(backend: Backend, tmp_path: Path):
    network = SequentialArrayNetwork((4,), MULTICLASS, SGD(), backend=backend)
    with pytest.raises(NotImplementedError, match="format 2"):
        network.save(str(tmp_path / "model.json"))
    for cls in (
        SequentialVectorizedMultiClassBackpropClassifierNetwork,
        SequentialRustArrayMultiClassBackpropClassifierNetwork,
        SequentialArrayBackpropClassifierNetwork,
        SequentialRustArrayBackpropClassifierNetwork,
    ):
        with pytest.raises(NotImplementedError, match="format 2"):
            cls.load(str(tmp_path / "model.json"))


def test_a_combination_no_preset_has_trains(backend: Backend):
    # conv, pool, a ReLU layer, a dropout layer and a softmax output, under Adam: every weight
    # moves, and the network still classifies
    layers = [
        Conv(3, 2),
        Pool(2),
        Dense(6, activation="relu"),
        Dense(5, dropout=0.2),
        Dense(3, output=True, activation="softmax", loss="cross_entropy"),
    ]
    backend.seed(0)
    network = SequentialArrayNetwork((8, 8, 1), layers, Adam(), backend=backend)
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
