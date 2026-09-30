"""
Every legacy save envelope keeps loading (the composable-layers workplan, stage 0), and so does
every format-2 fixture (stage 5): each class loads its committed fixture
(tests/saved_model_fixtures.py), and the loaded network holds the saved weights and hyperparameters
and predicts what the saved network did, by bits. A legacy file loads with fresh optimizer state,
and a format-2 file with the saved state, by bits.

The pure-Python and Rust predictions are compared with the ones stored beside the file. numpy's
go through BLAS, which can round differently on another machine (as the golden run's files), so a
numpy network is compared with a network of its class restored from the stored snapshot on this
machine instead. Its weights are still compared with the stored bits.
"""

import json
from typing import Any

import pytest

from tests.saved_model_fixtures import FIXTURE_DIR, FIXTURES, MODEL_CLASSES, bits, from_bits, optimizer_bits, outputs


def _expected(name: str) -> dict[str, Any]:
    with open(FIXTURE_DIR / f"{name}.expected.json") as f:
        return json.load(f)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_legacy_file_loads(name: str) -> None:
    fixture = FIXTURES[name]
    expected = _expected(name)

    loaded = MODEL_CLASSES[name].load(str(FIXTURE_DIR / f"{name}.json"))

    assert bits(loaded.snapshot()) == expected["snapshot"]
    if fixture.format2:
        assert optimizer_bits(loaded) == expected["optimizer_state"]
    else:
        # a legacy envelope has no optimizer state: training on starts it afresh
        for network in getattr(loaded, "classifiers", [loaded]):
            assert optimizer_bits(network) == [0, []]
    for hyperparameter, value in expected["hyperparameters"].items():
        assert getattr(loaded, hyperparameter) == value, hyperparameter

    states = [tuple(state) for state in from_bits(expected["states"])]
    if fixture.implementation == "numpy":
        reference = fixture.build()
        reference.restore(from_bits(expected["snapshot"]))
        expected_outputs = outputs(reference, fixture.predict, states)
    else:
        expected_outputs = expected["outputs"]
    assert outputs(loaded, fixture.predict, states) == expected_outputs


def test_every_saveable_class_has_a_fixture() -> None:
    # the networks, not the shape mixins and bases that define save for them
    saveable = {name for name, cls in MODEL_CLASSES.items() if name.endswith("Network") and hasattr(cls, "save")}
    assert saveable == set(FIXTURES)
    for name in FIXTURES:
        assert (FIXTURE_DIR / f"{name}.json").exists(), name
