"""
indrajala_ml.pcg64 is numpy's default_rng in pure Python, bit for bit (the RNG generators workplan,
D5): SeedSequence's pools, generate_state and spawns, PCG64's doubles and uniforms, and numpy's
bit_generator.state, which moves between numpy, the crate and this module and continues
identically. rust/tests/test_random_pcg64_parity.py checks the crate's port the same way.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from typing import Any, cast

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.pcg64 import Pcg64Generator, SeedSequence, default_rng

SEEDS: list[Any] = [
    0,
    1,
    7,
    2**32 - 1,
    2**32,
    2**64 + 1,
    2**127,
    2**128 + 3,
    2**200 + 5,
    [],
    [0],
    [1, 2, 3],
    (2**40, 1),
    range(5),
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [True, 2],
]
SEED_IDS = [repr(seed) for seed in SEEDS]


def doubles(rng: Pcg64Generator, n: int) -> bytes:
    return np.array([rng.random() for _ in range(n)], dtype=np.float64).tobytes()


@pytest.mark.parametrize("seed", SEEDS, ids=SEED_IDS)
def test_seed_sequence_matches_numpy(seed: Any):
    reference = np.random.SeedSequence(seed)
    ours = SeedSequence(seed)
    assert ours.pool == reference.pool.tolist()
    assert ours.generate_state(9) == reference.generate_state(9).tolist()
    assert ours.generate_state(5, np.uint64) == reference.generate_state(5, np.uint64).tolist()
    assert ours.generate_state(5, "uint64") == reference.generate_state(5, np.dtype("uint64")).tolist()


@pytest.mark.parametrize("spawn_key", [(0,), (1, 2), (2**40,), [3]], ids=repr)
def test_an_explicit_spawn_key_matches_numpy(spawn_key: Any):
    for entropy in [5, 2**100, [1, 2, 3, 4, 5]]:
        reference = np.random.SeedSequence(entropy, spawn_key=spawn_key)
        ours = SeedSequence(entropy, spawn_key=spawn_key)
        assert ours.pool == reference.pool.tolist()
        assert ours.spawn_key == reference.spawn_key


def test_spawned_children_and_grandchildren_match_numpy():
    reference = np.random.SeedSequence(2024)
    ours = SeedSequence(2024)
    pairs = list(zip(reference.spawn(3) + reference.spawn(2), ours.spawn(3) + ours.spawn(2), strict=True))
    assert ours.n_children_spawned == reference.n_children_spawned == 5
    for numpy_child, our_child in pairs:
        assert our_child.spawn_key == numpy_child.spawn_key
        assert our_child.pool == numpy_child.pool.tolist()
        for numpy_grandchild, our_grandchild in zip(numpy_child.spawn(2), our_child.spawn(2), strict=True):
            assert our_grandchild.pool == numpy_grandchild.pool.tolist()
        assert doubles(default_rng(our_child), 20) == np.random.default_rng(numpy_child).random(20).tobytes()


@pytest.mark.parametrize("seed", SEEDS, ids=SEED_IDS)
def test_default_rng_starts_in_numpys_state_and_draws_its_doubles(seed: Any):
    ours = default_rng(seed)
    reference = np.random.default_rng(seed)
    assert ours.state == reference.bit_generator.state
    assert doubles(ours, 300) == reference.random(300).tobytes()
    assert ours.state == reference.bit_generator.state


def test_seeds_whose_state_crosses_the_top_bit_match_numpy():
    # XSL-RR's rotation reads the state's top six bits; both values of the top bit must be covered
    top_bits: set[int] = set()
    for seed in range(64):
        ours = default_rng(seed)
        reference = np.random.default_rng(seed)
        for _ in range(4):
            top_bits.add(ours.state["state"]["state"] >> 127)
            assert doubles(ours, 25) == reference.random(25).tobytes()
    assert top_bits == {0, 1}


@pytest.mark.parametrize(("low", "high"), [(0.0, 1.0), (-0.25, 0.75), (5.0, 5.0), (-1e300, 1e300)])
def test_uniform_matches_numpy(low: float, high: float):
    ours = default_rng(7)
    values = np.array([ours.uniform(low, high) for _ in range(500)], dtype=np.float64)
    assert values.tobytes() == np.random.default_rng(7).uniform(low, high, 500).tobytes()


@pytest.mark.parametrize(
    ("low", "high"),
    [(0.0, float("inf")), (float("inf"), 0.0), (float("nan"), 1.0), (-1e308, 1e308), (1.0, 0.0)],
)
def test_uniform_raises_numpys_errors_on_a_non_finite_or_negative_range(low: float, high: float):
    with pytest.raises((OverflowError, ValueError)) as numpy_error:
        np.random.default_rng(7).uniform(low, high)
    with pytest.raises((OverflowError, ValueError)) as our_error:
        default_rng(7).uniform(low, high)
    assert our_error.type is numpy_error.type
    assert str(our_error.value) == str(numpy_error.value)


def test_interleaved_random_and_uniform_carry_the_position():
    ours = default_rng(2024)
    reference = np.random.default_rng(2024)
    for _ in range(50):
        assert ours.uniform(-1.0, 1.0) == reference.uniform(-1.0, 1.0)
        assert ours.random() == reference.random()
        assert (ours.random() >= 0.3) == (reference.random() >= 0.3)


def test_a_state_moves_between_numpy_the_crate_and_pure_python():
    reference = np.random.default_rng(99)
    reference.random(37)
    ours = default_rng()
    ours.state = dict(reference.bit_generator.state)
    assert doubles(ours, 40) == reference.random(40).tobytes()
    crate = pa.default_rng()
    crate.state = ours.state
    assert np.array(crate.random(40).tolist()).tobytes() == reference.random(40).tobytes()
    ours.state = crate.state
    assert doubles(ours, 40) == reference.random(40).tobytes()


def test_numpys_and_the_crates_seed_sequences_seed_it_as_numpy_does():
    assert default_rng(np.random.SeedSequence(5)).state == np.random.default_rng(5).bit_generator.state
    assert default_rng(pa.SeedSequence(5)).state == np.random.default_rng(5).bit_generator.state


def test_the_buffered_32_bit_fields_round_trip_unchanged():
    reference = np.random.default_rng(4)
    reference.integers(0, 2**32, dtype=np.uint32)
    state = dict(reference.bit_generator.state)
    assert state["has_uint32"] == 1
    ours = default_rng()
    ours.state = state
    assert ours.state == state
    assert doubles(ours, 10) == reference.random(10).tobytes()


REJECTED_STATES: list[tuple[str, Any, type[Exception]]] = [
    ("not a dict", 5, TypeError),
    ("another generator", {"bit_generator": "MT19937"}, ValueError),
    ("no inc", {"bit_generator": "PCG64", "state": {"state": 1}}, KeyError),
    ("state over 128 bits", {"bit_generator": "PCG64", "state": {"state": 2**128, "inc": 1}}, OverflowError),
    ("negative inc", {"bit_generator": "PCG64", "state": {"state": 1, "inc": -1}}, OverflowError),
]


@pytest.mark.parametrize(
    ("state", "error"),
    [(state, error) for _label, state, error in REJECTED_STATES],
    ids=[label for label, _state, _error in REJECTED_STATES],
)
def test_a_rejected_state_raises_numpys_exception_type_and_changes_nothing(state: Any, error: type[Exception]):
    full = {"has_uint32": 0, "uinteger": 0} | cast(dict[str, Any], state) if isinstance(state, dict) else state
    reference = np.random.default_rng(6)
    ours = default_rng(6)
    with pytest.raises(error):
        reference.bit_generator.state = full
    with pytest.raises(error):
        ours.state = full
    assert ours.state == reference.bit_generator.state


@pytest.mark.parametrize("seed", [-1, 1.5, [1.0], [-1], [[1, 2]], "12"], ids=repr)
def test_a_seed_numpy_rejects_raises_numpys_exception_type(seed: Any):
    with pytest.raises((TypeError, ValueError)) as numpy_error:
        np.random.default_rng(seed)
    with pytest.raises((TypeError, ValueError)) as our_error:
        default_rng(seed)
    assert our_error.type is numpy_error.type


def test_default_rng_returns_a_generator_as_it_is():
    rng = default_rng(3)
    assert default_rng(rng) is rng


def test_none_draws_fresh_entropy_in_each_process():
    assert default_rng().state != default_rng().state
    entropy = SeedSequence().entropy
    assert 0 <= entropy < 2**128

    def draw() -> str:
        statement = "from indrajala_ml.pcg64 import default_rng; print(default_rng().random())"
        return subprocess.run([sys.executable, "-c", statement], capture_output=True, text=True, check=True).stdout

    assert draw() != draw()


def test_pickle_carries_the_state():
    rng = default_rng(8)
    rng.random()
    clone = pickle.loads(pickle.dumps(rng))
    assert clone.state == rng.state
    assert clone.random() == rng.random()
