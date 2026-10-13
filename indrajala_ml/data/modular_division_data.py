"""
Modular division of two base-3 numbers read most significant digit first, and the catalogue of
properties a network that divides must have (docs/window-recurrent-claims-workplan.md, E7 and
D14-D18).

A case is a pair (a, b) of n-digit base-3 numbers, 0 <= a, b < 3^n, read a column at a time from
the most significant digit, a column's token 3 a_i + b_i (9 values). After each column the answer
is the running quotient (A * B^-1) mod p of the prefixes A and B read so far, or UNDEFINED (class p)
while B mod p is 0: n answers, the last the quotient of a and b. The state (A mod p, B mod p) is
all a reader needs, so the task is a finite transducer of p^2 states (window_tasks' modular
machine); its answer is unchanged by trailing zero columns (both prefixes times 3, and 3 is
invertible mod p) and by leading ones (the state stays (0, 0), undefined).

A property is a statement, a generator of cases and a coverage, every case verified the same way:
the n answers are the running quotients. A Solver is what a property is verified against: a
batch of pairs to their n answers each. Every property's cases are drawn from a generator seeded
by p, the property and n alone, so every network meets the same cases; the training mixture draws
from other seeds and never draws a held-out pair (D8's tenth, by hash).
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from random import Random

from indrajala_ml.data.addition_data import (
    BASE,
    EXHAUSTIVE_LIMIT,
    FAILURES_KEPT,
    HELD_OUT_TENTHS,
    PER_STRATUM,
    SAMPLED,
    digits,
    doubling_distances,
    splitmix64,
)

PRIMES = (5, 7)  # D16: 25 and 49 states

Pair = tuple[int, int]
Answers = tuple[int, ...]
# a batch of pairs to their n answers each; a network's encoder, forward and decoder
Solver = Callable[[Sequence[Pair]], list[Answers]]


def undefined(p: int) -> int:
    """The answer class while B mod p is 0."""
    return p


def column_tokens(pair: Pair, n: int) -> tuple[int, ...]:
    """One token a column, 3 a_i + b_i, most significant first: n of them."""
    a, b = digits(pair[0], n), digits(pair[1], n)
    return tuple(BASE * a[t] + b[t] for t in reversed(range(n)))


def quotient(A: int, B: int, p: int) -> int:
    """(A * B^-1) mod p, or undefined(p) when B mod p is 0."""
    return undefined(p) if B % p == 0 else A * pow(B, -1, p) % p


def answers(pair: Pair, n: int, p: int) -> Answers:
    """The running quotient after each of the n columns, most significant first."""
    a = b = 0  # the prefixes mod p
    result: list[int] = []
    for token in column_tokens(pair, n):
        a, b = (BASE * a + token // BASE) % p, (BASE * b + token % BASE) % p
        result.append(quotient(a, b, p))
    return tuple(result)


def pair_hash(pair: Pair, n: int) -> int:
    """A 64-bit hash of a pair at n (order matters: a / b isn't b / a)."""
    return splitmix64((pair[0] * BASE**n + pair[1]) * 64 + n)


def held_out(pair: Pair, n: int) -> bool:
    """D8: whether a pair is in the held-out tenth."""
    return pair_hash(pair, n) % 10 < HELD_OUT_TENTHS


def uniform_pair(n: int, rng: Random) -> Pair:
    return (rng.randrange(BASE**n), rng.randrange(BASE**n))


def zero_prefix_pair(n: int, rng: Random, length: int | None = None) -> Pair:
    """
    A pair whose b starts with length zero digits (uniform in 1..n when None), its next digit
    nonzero: the answer undefined for that many columns at least.
    """
    if length is None:
        length = rng.randrange(1, n + 1)
    rest = n - length
    b = 0 if rest == 0 else rng.randrange(1, BASE) * BASE ** (rest - 1) + rng.randrange(BASE ** (rest - 1))
    return (rng.randrange(BASE**n), b)


def zero_dividend_pair(n: int, rng: Random) -> Pair:
    """a = 0: every defined answer is 0."""
    return (0, rng.randrange(BASE**n))


_KINDS = (uniform_pair, zero_prefix_pair, zero_dividend_pair)


def training_pair(kind: int, n: int, rng: Random, hold_out: bool = True) -> Pair:
    """
    One pair of the training mixture (D18): kind 0 uniform, 1 a zero prefix of b, 2 a zero a (kind
    taken mod 3); a held-out pair is drawn again, unless not hold_out.
    """
    draw = _KINDS[kind % 3]
    pair = draw(n, rng)
    while hold_out and held_out(pair, n):
        pair = draw(n, rng)
    return pair


@dataclass(frozen=True)
class Case:
    """One case of a property: a pair and its stratum."""

    pair: Pair
    stratum: int | None = None


@dataclass(frozen=True)
class Property:
    id: str
    statement: str
    generate: Callable[[int, Random], list[Case]]
    coverage: str
    p: int


@dataclass
class PropertyResult:
    id: str
    cases: int
    passed: int
    # stratum: (passed, cases)
    strata: dict[int, tuple[int, int]] = field(default_factory=dict[int, tuple[int, int]])
    failures: list[Case] = field(default_factory=list[Case])  # the first FAILURES_KEPT

    @property
    def holds(self) -> bool:
        return self.passed == self.cases

    def reach(self) -> int:
        """S2's L*: the largest zero prefix L with every case at strata up to L right (0 when the first fails)."""
        reach = 0
        for stratum in sorted(self.strata):
            passed, cases = self.strata[stratum]
            if passed < cases:
                break
            reach = stratum
        return reach


def _all_pairs(n: int) -> list[Case]:
    r = range(BASE**n)
    return [Case((a, b)) for a in r for b in r]


def _divides(n: int, rng: Random, sampled: int) -> list[Case]:
    if BASE ** (2 * n) <= EXHAUSTIVE_LIMIT:
        return _all_pairs(n)
    return [Case(uniform_pair(n, rng)) for _ in range(sampled)]


def _held_out(n: int, rng: Random, sampled: int) -> list[Case]:
    if BASE ** (2 * n) <= EXHAUSTIVE_LIMIT:
        return [c for c in _all_pairs(n) if held_out(c.pair, n)]
    cases: list[Case] = []
    while len(cases) < sampled:
        pair = uniform_pair(n, rng)
        if held_out(pair, n):
            cases.append(Case(pair))
    return cases


def _every_state(n: int, rng: Random, per_stratum: int, p: int) -> list[Case]:
    # uniform pairs bucketed by their final state A * p + B (mod p each) until every state the
    # numbers of n digits reach has per_stratum
    residues = sorted({x % p for x in range(min(BASE**n, p))})
    want = {A * p + B: per_stratum for A, B in itertools.product(residues, repeat=2)}
    cases: list[Case] = []
    while want:
        pair = uniform_pair(n, rng)
        state = pair[0] % p * p + pair[1] % p
        if want.get(state, 0) > 0:
            cases.append(Case(pair, stratum=state))
            want[state] -= 1
            if want[state] == 0:
                del want[state]
    return cases


def _undefined_run(n: int, rng: Random, per_stratum: int, doubling: bool = False) -> list[Case]:
    lengths = doubling_distances(n) if doubling else range(1, n + 1)
    return [Case(zero_prefix_pair(n, rng, length), stratum=length) for length in lengths for _ in range(per_stratum)]


def catalogue(
    p: int, sampled: int = SAMPLED, per_stratum: int = PER_STRATUM, doubling: bool = False
) -> tuple[Property, ...]:
    """
    The catalogue (D18), its samples sized: sampled cases for U1 and U2 past the exhaustive limit,
    per_stratum per S1 final state and per S2 prefix length, S2 at doubling_distances only when
    doubling (the long tests, D9).
    """
    sized = functools.partial
    return (
        Property(
            "U1",
            "divides: every column's answer is the running quotient",
            sized(_divides, sampled=sampled),
            f"every pair, or {sampled:,} sampled",
            p,
        ),
        Property(
            "U2",
            "divides on the held-out tenth (D8)",
            sized(_held_out, sampled=sampled),
            f"the whole tenth, or {sampled:,} sampled",
            p,
        ),
        Property(
            "S1",
            "every final state: (a mod p, b mod p), each reachable one",
            sized(_every_state, per_stratum=per_stratum, p=p),
            f"{per_stratum:,} per state",
            p,
        ),
        Property(
            "S2",
            "undefined for a run: b's first L digits zero, then defined",
            sized(_undefined_run, per_stratum=per_stratum, doubling=doubling),
            f"{per_stratum:,} per L in " + ("1, 2, 4, ..., n" if doubling else "1..n"),
            p,
        ),
    )


def property_cases(prop: Property, n: int) -> list[Case]:
    """A property's cases at n: the same for every network and every run."""
    return prop.generate(n, Random(f"modular-division-{prop.p}-{prop.id}-{n}"))


@functools.cache
def screen_cases(prop: Property, n: int, limit: int) -> list[Case]:
    """Every k-th of a property's cases, k the smallest that leaves at most limit; held per process."""
    cases = property_cases(prop, n)
    return cases[:: -(-len(cases) // limit)] if len(cases) > limit else cases


def evaluate(f: Solver, n: int, properties: Sequence[Property], limit: int | None = None) -> list[PropertyResult]:
    """Each property verified against f; limit takes every k-th case, as addition_data's screen."""
    results: list[PropertyResult] = []
    for prop in properties:
        cases = property_cases(prop, n) if limit is None else screen_cases(prop, n, limit)
        got = f([case.pair for case in cases])
        result = PropertyResult(prop.id, len(cases), 0)
        for case, answer in zip(cases, got, strict=True):
            ok = answer == answers(case.pair, n, prop.p)
            result.passed += ok
            if case.stratum is not None:
                passed, count = result.strata.get(case.stratum, (0, 0))
                result.strata[case.stratum] = (passed + ok, count + 1)
            if not ok and len(result.failures) < FAILURES_KEPT:
                result.failures.append(case)
        results.append(result)
    return results
