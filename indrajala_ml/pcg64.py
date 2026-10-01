"""
numpy's default_rng in pure Python, for the pure-Python networks: Generator(PCG64(SeedSequence(seed))),
bit for bit, drawing one scalar at a time. numpy draws the same stream and the crate's pa.Generator
(rust/src/generator.rs) ports the same algorithm, so all three implementations draw from one
stream family and share one state layout, numpy's bit_generator.state
(the RNG generators workplan, D5). Backend-free: no numpy import.

- SeedSequence(entropy, spawn_key=()) hashes the entropy (an int of any size, a sequence of ints,
  or 128 bits of OS entropy for None) and the spawn key into a pool of four 32-bit words, and
  expands the pool into as many words as asked (generate_state).
- PCG64 (O'Neill 2014, XSL-RR 128/64) seeds its state and increment from 256 of those bits, steps
  a 128-bit LCG and outputs 64 bits per step.
- random() is (next_uint64 >> 11) * 2^-53; uniform(low, high) is low + (high - low) * random().
"""

from __future__ import annotations

import math
import operator
import secrets
from collections.abc import Iterable
from typing import Any, Protocol, cast, runtime_checkable

MASK_32 = 0xFFFFFFFF
MASK_64 = 0xFFFFFFFFFFFFFFFF
MASK_128 = (1 << 128) - 1

POOL_SIZE = 4
INIT_A = 0x43B0D7E5
MULT_A = 0x931E8875
INIT_B = 0x8B51F9DD
MULT_B = 0x58F38DED
MIX_MULT_L = 0xCA01F9DD
MIX_MULT_R = 0x4973F715
XSHIFT = 16

# PCG_DEFAULT_MULTIPLIER_128
PCG_MULTIPLIER = (2549297995355413924 << 64) | 4865540595714422341


def _int_to_words(value: Any) -> list[int]:
    """A non-negative integer as 32-bit words, lowest first ([0] for 0)."""
    n = operator.index(value)
    if n < 0:
        raise ValueError("expected non-negative integer")
    words = [n & MASK_32]
    n >>= 32
    while n:
        words.append(n & MASK_32)
        n >>= 32
    return words


def _coerce_to_words(value: Any) -> list[int]:
    """An int's words, or every element's words of a sequence of ints, concatenated."""
    if isinstance(value, float):
        raise TypeError("seed must be integer")
    try:
        return _int_to_words(value)
    except TypeError:
        pass
    words: list[int] = []
    for item in value:
        if isinstance(item, float):
            raise TypeError("seed must be integer")
        words.extend(_int_to_words(item))
    return words


def _hashmix(value: int, hash_const: list[int]) -> int:
    value ^= hash_const[0]
    hash_const[0] = (hash_const[0] * MULT_A) & MASK_32
    value = (value * hash_const[0]) & MASK_32
    return value ^ (value >> XSHIFT)


def _mix(x: int, y: int) -> int:
    result = (MIX_MULT_L * x - MIX_MULT_R * y) & MASK_32
    return result ^ (result >> XSHIFT)


def _mix_entropy(entropy: list[int]) -> list[int]:
    hash_const = [INIT_A]
    pool = [_hashmix(entropy[i] if i < len(entropy) else 0, hash_const) for i in range(POOL_SIZE)]
    for i_src in range(POOL_SIZE):
        for i_dst in range(POOL_SIZE):
            if i_src != i_dst:
                pool[i_dst] = _mix(pool[i_dst], _hashmix(pool[i_src], hash_const))
    for word in entropy[POOL_SIZE:]:
        for i_dst in range(POOL_SIZE):
            pool[i_dst] = _mix(pool[i_dst], _hashmix(word, hash_const))
    return pool


@runtime_checkable
class SeedSequenceLike(Protocol):
    """What seeds a generator: numpy's SeedSequence, the crate's, or this module's."""

    def generate_state(self, n_words: int, dtype: Any = ...) -> Any: ...


class SeedSequence:
    """
    np.random.SeedSequence(entropy, spawn_key=()) with numpy's pool of four words. entropy is a
    non-negative int, a sequence of them, or None for 128 bits of OS entropy (secrets.randbits,
    as numpy does). numpy's seed strings aren't taken.
    """

    def __init__(self, entropy: int | Iterable[int] | None = None, *, spawn_key: Iterable[int] = ()) -> None:
        self.entropy: Any = secrets.randbits(32 * POOL_SIZE) if entropy is None else entropy
        self.spawn_key = tuple(spawn_key)
        self.pool_size = POOL_SIZE
        self.n_children_spawned = 0
        run_entropy = _coerce_to_words(self.entropy)
        spawn_entropy = _coerce_to_words(self.spawn_key)
        if spawn_entropy and len(run_entropy) < POOL_SIZE:
            # a spawn key pads the run entropy, so its words never stand in for missing entropy
            run_entropy += [0] * (POOL_SIZE - len(run_entropy))
        self.pool = _mix_entropy(run_entropy + spawn_entropy)

    def generate_state(self, n_words: int, dtype: Any = "uint32") -> list[int]:
        """
        The pool expanded into n_words words. dtype is uint32 or uint64, as a name or numpy's type
        or dtype; a uint64 word is two uint32 words, low first.
        """
        name = dtype if isinstance(dtype, str) else getattr(dtype, "__name__", None) or dtype.name
        if name not in ("uint32", "uint64"):
            raise ValueError("only support uint32 or uint64")
        n_32 = n_words * 2 if name == "uint64" else n_words
        hash_const = INIT_B
        words: list[int] = []
        for i in range(n_32):
            value = self.pool[i % POOL_SIZE] ^ hash_const
            hash_const = (hash_const * MULT_B) & MASK_32
            value = (value * hash_const) & MASK_32
            words.append(value ^ (value >> XSHIFT))
        if name == "uint64":
            return [words[i] | words[i + 1] << 32 for i in range(0, n_32, 2)]
        return words

    def spawn(self, n_children: int) -> list[SeedSequence]:
        """Children with spawn keys spawn_key + (i,), i counting on from earlier spawns."""
        first = self.n_children_spawned
        self.n_children_spawned += n_children
        return [SeedSequence(self.entropy, spawn_key=(*self.spawn_key, i)) for i in range(first, first + n_children)]

    def __repr__(self) -> str:
        spawn_key = f", spawn_key={self.spawn_key!r}" if self.spawn_key else ""
        return f"SeedSequence(entropy={self.entropy!r}{spawn_key})"


class Pcg64Generator:
    """
    numpy's Generator(PCG64(seed_sequence)), one scalar per call. state is numpy's
    bit_generator.state dict, so it moves to and from numpy's and the crate's generators.
    """

    def __init__(self, seed_sequence: SeedSequenceLike) -> None:
        seed = [int(word) for word in seed_sequence.generate_state(4, "uint64")]
        # pcg64_set_seed, then pcg_setseq_128_srandom_r
        self._inc = ((seed[2] << 64 | seed[3]) << 1 | 1) & MASK_128
        self._state = 0
        self._step()
        self._state = (self._state + (seed[0] << 64 | seed[1])) & MASK_128
        self._step()
        # numpy's buffered half of a 64-bit draw for 32-bit draws, which nothing here makes; kept
        # so a state round-trips unchanged
        self._has_uint32 = 0
        self._uinteger = 0

    def _step(self) -> None:
        self._state = (self._state * PCG_MULTIPLIER + self._inc) & MASK_128

    def next_uint64(self) -> int:
        """One step, then XSL-RR: the state's halves xored, rotated right by its top six bits."""
        self._step()
        folded = ((self._state >> 64) ^ self._state) & MASK_64
        rotation = self._state >> 122
        return (folded >> rotation | folded << (64 - rotation)) & MASK_64

    def random(self) -> float:
        """A double in [0, 1): the top 53 bits of one draw."""
        return (self.next_uint64() >> 11) * (1.0 / 9007199254740992.0)

    def uniform(self, low: float, high: float) -> float:
        """
        low + (high - low) * random(). A non-finite range raises numpy's OverflowError and a
        negative one its ValueError.
        """
        span = high - low
        if not math.isfinite(span):
            raise OverflowError("high - low range exceeds valid bounds")
        if span < 0:
            raise ValueError("high - low < 0")
        return low + span * self.random()

    @property
    def state(self) -> dict[str, Any]:
        return {
            "bit_generator": "PCG64",
            "state": {"state": self._state, "inc": self._inc},
            "has_uint32": self._has_uint32,
            "uinteger": self._uinteger,
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        """numpy's state dict, checked as numpy's setter checks it; a rejected one changes nothing."""
        if not isinstance(cast(object, value), dict):  # for callers outside the type checker
            raise TypeError("state must be a dict")
        if value.get("bit_generator", "") != "PCG64":
            raise ValueError("state must be for a PCG64 RNG")
        state = operator.index(value["state"]["state"])
        inc = operator.index(value["state"]["inc"])
        has_uint32 = operator.index(value["has_uint32"])
        uinteger = operator.index(value["uinteger"])
        for field, bound in ((state, MASK_128), (inc, MASK_128), (uinteger, MASK_32)):
            if not 0 <= field <= bound:
                raise OverflowError(f"Python integer {field} out of bounds")
        self._state, self._inc, self._has_uint32, self._uinteger = state, inc, has_uint32, uinteger

    def __repr__(self) -> str:
        return "Pcg64Generator(PCG64)"


def default_rng(seed: int | Iterable[int] | SeedSequenceLike | Pcg64Generator | None = None) -> Pcg64Generator:
    """
    np.random.default_rng(seed): a Pcg64Generator is returned as it is; a seed sequence (this
    module's, numpy's or the crate's) seeds a new one; anything else seeds one through SeedSequence.
    """
    if isinstance(seed, Pcg64Generator):
        return seed
    if isinstance(seed, SeedSequenceLike):
        return Pcg64Generator(seed)
    return Pcg64Generator(SeedSequence(seed))


def generator_state(rng: Any) -> dict[str, Any]:
    """
    rng's state as numpy's bit_generator.state dict, for numpy's Generator, the crate's and this
    module's alike: numpy's Generator keeps it on its bit generator, the other two on themselves.
    """
    return getattr(rng, "bit_generator", rng).state


def set_generator_state(rng: Any, state: dict[str, Any]) -> None:
    """Sets rng's state, as generator_state gives it, in place: whatever holds rng draws on from it."""
    getattr(rng, "bit_generator", rng).state = state
