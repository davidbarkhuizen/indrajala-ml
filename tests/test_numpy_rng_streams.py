"""
numpy's random streams are reproducible from outside numpy, bit for bit (docs/rng-audit.md). A
pure-Python MT19937, seeded as np.random.seed(int) seeds it, reproduces the legacy stream that
randomize() (np.random.uniform) and DropoutArrayLayer (np.random.random >= p) draw from, and the
PCG64 Generator's floats are the crate's own (raw >> 11) * 2^-53. np.random.seed's three paths
(docs/rng-numpy-parity-workplan.md, "Seeding, exactly") are pinned too: init_genrand for anything
operator.index takes (after squeeze), init_by_array for other sequences, and the rejections. The
reference here is the oracle a seedable Rust generator is checked against.
"""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

MASK_32 = 0xFFFFFFFF


class MersenneTwister:
    """MT19937 (Matsumoto & Nishimura 1998), init_genrand seeding, numpy's legacy random_double."""

    def __init__(self, seed: int) -> None:
        self.state = [seed & MASK_32]
        for i in range(1, 624):
            previous = self.state[-1]
            self.state.append((1812433253 * (previous ^ (previous >> 30)) + i) & MASK_32)
        self.index = 624

    @classmethod
    def from_key(cls, key: Sequence[int]) -> "MersenneTwister":
        """init_by_array (mt19937ar.c), numpy's seeding for a sequence seed."""
        mt = cls(19650218)
        state = mt.state
        i, j = 1, 0
        for _ in range(max(624, len(key))):
            previous = state[i - 1]
            state[i] = ((state[i] ^ ((previous ^ (previous >> 30)) * 1664525)) + key[j] + j) & MASK_32
            i, j = i + 1, (j + 1) % len(key)
            if i >= 624:
                state[0] = state[623]
                i = 1
        for _ in range(623):
            previous = state[i - 1]
            state[i] = ((state[i] ^ ((previous ^ (previous >> 30)) * 1566083941)) - i) & MASK_32
            i += 1
            if i >= 624:
                state[0] = state[623]
                i = 1
        state[0] = 0x80000000  # non-zero initial array
        return mt

    def _twist(self) -> None:
        state = self.state
        for k in range(624):
            y = (state[k] & 0x80000000) | (state[(k + 1) % 624] & 0x7FFFFFFF)
            state[k] = state[(k + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
        self.index = 0

    def next_u32(self) -> int:
        if self.index >= 624:
            self._twist()
        y = self.state[self.index]
        self.index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        return (y ^ (y >> 18)) & MASK_32

    def next_double(self) -> float:
        # 27 + 26 bits of two draws: 53 random mantissa bits
        a = self.next_u32() >> 5
        b = self.next_u32() >> 6
        return (a * 67108864.0 + b) / 9007199254740992.0


@pytest.mark.parametrize("seed", [0, 1, 42, 2**32 - 1])
def test_mersenne_twister_reproduces_the_legacy_stream_randomize_and_dropout_draw(seed: int):
    # randomize()'s W then b, then a dropout mask; 50 * 20 + 20 + 8 * 20 doubles cross several
    # 312-double twist boundaries
    np.random.seed(seed)
    W = np.random.uniform(-0.25, 0.25, size=(50, 20))
    b = np.random.uniform(-0.25, 0.25, size=(20,))
    mask = np.random.random((8, 20)) >= 0.3

    mt = MersenneTwister(seed)
    W_reference = np.array([[-0.25 + 0.5 * mt.next_double() for _ in range(20)] for _ in range(50)])
    b_reference = np.array([-0.25 + 0.5 * mt.next_double() for _ in range(20)])
    mask_reference = np.array([[mt.next_double() >= 0.3 for _ in range(20)] for _ in range(8)])

    assert np.array_equal(W.view(np.uint64), W_reference.view(np.uint64))
    assert np.array_equal(b.view(np.uint64), b_reference.view(np.uint64))
    assert np.array_equal(mask, mask_reference)


def test_pcg64_generator_floats_are_the_crates_top_53_bits_formula():
    generator = np.random.default_rng(7)
    raw = np.random.default_rng(7).bit_generator.random_raw(1000)
    unit = (raw >> np.uint64(11)).astype(np.float64) * 2.0**-53
    assert np.array_equal(generator.uniform(-0.3, 0.3, 1000), -0.3 + 0.6 * unit)


def _same_stream(seed: Any, reference: MersenneTwister) -> bool:
    np.random.seed(seed)
    return all(float(np.random.random()) == reference.next_double() for _ in range(700))


@pytest.mark.parametrize(
    "seed, int_seed",
    [(True, 1), (np.uint64(7), 7), (np.int8(3), 3), (np.array([5]), 5), (np.array([[5]]), 5)],
)
def test_anything_operator_index_takes_after_squeeze_seeds_by_init_genrand(seed: Any, int_seed: int):
    assert _same_stream(seed, MersenneTwister(int_seed))


@pytest.mark.parametrize(
    "seed, key",
    [
        ([5], [5]),  # a list has no squeeze: init_by_array, a different stream from seed(5)
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2), [1, 2]),
        (range(3), [0, 1, 2]),
        ([True, 2], [1, 2]),
        (np.array([1, 2], dtype=np.uint32), [1, 2]),
        (np.array([1, 2], dtype=np.int8), [1, 2]),
        (list(range(700)), list(range(700))),  # a key longer than the 624-word state
    ],
)
def test_other_sequences_seed_by_init_by_array(seed: Any, key: list[int]):
    assert _same_stream(seed, MersenneTwister.from_key(key))


@pytest.mark.parametrize(
    "seed, error, message",
    [
        (-1, ValueError, "Seed must be between 0 and 2**32 - 1"),
        (2**32, ValueError, "Seed must be between 0 and 2**32 - 1"),
        ([1, 2**40], ValueError, "Seed must be between 0 and 2**32 - 1"),
        ([2**63 - 1], ValueError, "Seed must be between 0 and 2**32 - 1"),
        ([], ValueError, "Seed must be non-empty"),
        ([[1, 2], [3, 4]], ValueError, "Seed array must be 1-d"),
        (1.5, TypeError, None),
        ("3", TypeError, None),
        ([1.0], TypeError, None),
        (np.array([1, 2], dtype=np.uint64), TypeError, None),  # uint64 -> int64 isn't a safe cast
        ([2**63], TypeError, None),  # the list becomes uint64 before the range check
        ([2**64], TypeError, None),  # ... or object
    ],
)
def test_rejected_seeds(seed: Any, error: type[Exception], message: str | None):
    with pytest.raises(error) as raised:
        np.random.seed(seed)
    if message is not None:
        assert str(raised.value) == message


def _legacy_state() -> tuple[Any, ...]:
    # ("MT19937", key, position, has_gauss, cached_gaussian); typed as a union with the dict form
    return cast(tuple[Any, ...], np.random.get_state())


@pytest.mark.parametrize("doubles_drawn", [0, 5, 311])
def test_seed_none_sets_word_0_but_keeps_the_previous_position(doubles_drawn: int):
    # unlike the int and sequence paths, which reset the position to 624, None refills the key
    # from entropy and leaves the position alone
    np.random.seed(1)
    np.random.random(doubles_drawn)
    position_before = _legacy_state()[2]
    np.random.seed(None)
    _name, key, position, *_ = _legacy_state()
    assert key[0] == 0x80000000
    assert position == position_before
