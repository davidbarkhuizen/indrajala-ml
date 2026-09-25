"""
numpy's random streams are reproducible from outside numpy, bit for bit (docs/rng-audit.md). A
pure-Python MT19937, seeded as np.random.seed(int) seeds it, reproduces the legacy stream that
randomize() (np.random.uniform) and DropoutArrayLayer (np.random.random >= p) draw from, and the
PCG64 Generator's floats are the crate's own (raw >> 11) * 2^-53. The reference here is the oracle
a seedable Rust generator would be checked against.
"""

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
