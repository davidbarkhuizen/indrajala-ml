"""
Base-3 addition of three operands, and the catalogue of properties a network that adds must have
(the addition study workplan, docs/addition-study-workplan.md, Stage 2).

An addition is a triple (a, b, c) of n-digit base-3 operands, 0 <= a, b, c < 3^n; its sum, under
3^(n+1), has n + 1 digits. Two-operand addition is the case with a zero in one slot. Digits are
least significant first throughout. Column t's digit sum s_t = a_t + b_t + c_t is in 0..6; with the
carry in k_t (0..2) the sum's digit t is (s_t + k_t) mod 3 and the carry out (s_t + k_t) // 3. A
column sum of 0, 3 or 6 is a reset (its carry out is 0, 1 or 2 whatever comes in); 1, 2, 4 and 5
are dependent. The carry into column t is fixed by the nearest reset below it (or column 0's carry
in, 0) composed through the dependent columns between: their count is the carry's distance.

A property is a statement, a generator of cases, a verifier and a coverage (the workplan's
catalogue). An Adder is what a property is verified against: a candidate network's encoder,
forward pass and decoder, or a plain function, taking a batch of triples to their sums. evaluate()
runs the catalogue against one and shrinks each failure to a minimal failing case. Every
property's cases are drawn from a generator seeded by the property and n alone, so every candidate
meets the same cases. The training mixture draws from other seeds and never draws a held-out
triple (D8).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from random import Random

BASE = 3
COLUMN_SUMS = range(3 * (BASE - 1) + 1)  # 0..6
RESETS = (0, 3, 6)  # the column sums whose carry out doesn't depend on the carry in
DEPENDENT = (1, 2, 4, 5)
HELD_OUT_TENTHS = 1  # D8: a tenth of the triples, by hash
EXHAUSTIVE_LIMIT = 600_000  # a space of at most this many cases is tested whole
SAMPLED = 100_000  # the behavioural properties' sample, over a larger space
RELATION_SAMPLED = 10_000  # A1 and A2
PER_STRATUM = 1_000  # M3 per distance, M4 per final carry
PATTERN_CASES = 10_000  # M5: per carry pattern, max(1, this // 7^n) concrete triples
FAILURES_KEPT = 3  # shrunk failing cases kept per property

Triple = tuple[int, int, int]
# a batch of triples to their sums (each under 3^(n+1)); a network's encoder, forward and decoder
Adder = Callable[[Sequence[Triple]], list[int]]


def digits(x: int, n: int) -> tuple[int, ...]:
    """x's n base-3 digits, least significant first."""
    assert 0 <= x < BASE**n, f"{x} doesn't have {n} base-{BASE} digits"
    return tuple((x // BASE**t) % BASE for t in range(n))


def from_digits(ds: Sequence[int]) -> int:
    return sum(d * BASE**t for t, d in enumerate(ds))


def column_sums(triple: Triple, n: int) -> tuple[int, ...]:
    a, b, c = (digits(x, n) for x in triple)
    return tuple(a[t] + b[t] + c[t] for t in range(n))


def carries(sums: Sequence[int]) -> tuple[int, ...]:
    """The carry into each column 0..n (n + 1 of them, the last the sum's top digit)."""
    k = [0]
    for s in sums:
        k.append((s + k[-1]) // BASE)
    return tuple(k)


def distance(sums: Sequence[int], t: int) -> int:
    """The count of dependent columns directly below column t, down to a reset or column 0."""
    count = 0
    while t - 1 - count >= 0 and sums[t - 1 - count] not in RESETS:
        count += 1
    return count


def sum_digit(total: int, t: int) -> int:
    return (total // BASE**t) % BASE


def held_out(triple: Triple, n: int) -> bool:
    """D8: whether a triple is in the held-out tenth; every order of a triple alike."""
    x, y, z = sorted(triple)
    key = ((x * BASE**n + y) * BASE**n + z) * 64 + n
    return _splitmix64(key) % 10 < HELD_OUT_TENTHS


def _splitmix64(x: int) -> int:
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return x ^ (x >> 31)


# a column sum's digit triples: 1, 3, 6, 7, 6, 3, 1 of them
_COLUMN_DIGITS = {s: [d for d in itertools.product(range(BASE), repeat=3) if sum(d) == s] for s in COLUMN_SUMS}


def triple_of_sums(sums: Sequence[int], rng: Random) -> Triple:
    """A triple with these column sums, each column's digits uniform among those with its sum."""
    columns = [rng.choice(_COLUMN_DIGITS[s]) for s in sums]
    return (
        from_digits([column[0] for column in columns]),
        from_digits([column[1] for column in columns]),
        from_digits([column[2] for column in columns]),
    )


def uniform_triple(n: int, rng: Random) -> Triple:
    return (rng.randrange(BASE**n), rng.randrange(BASE**n), rng.randrange(BASE**n))


def pattern_triple(n: int, rng: Random) -> Triple:
    """A triple whose carry pattern (column sums) is uniform over the 7^n."""
    return triple_of_sums([rng.choice(COLUMN_SUMS) for _ in range(n)], rng)


def two_operand_triple(n: int, rng: Random) -> Triple:
    """A uniform pair, with the zero in a uniform slot."""
    pair = [rng.randrange(BASE**n), rng.randrange(BASE**n)]
    pair.insert(rng.randrange(3), 0)
    return (pair[0], pair[1], pair[2])


def training_triples(n: int, count: int, rng: Random) -> list[Triple]:
    """
    D7's mixture: a third uniform triples, a third uniform over carry patterns, a third two-operand
    cases, in turn; a held-out triple (D8) is drawn again.
    """
    kinds = (uniform_triple, pattern_triple, two_operand_triple)
    triples: list[Triple] = []
    for i in range(count):
        triple = kinds[i % 3](n, rng)
        while held_out(triple, n):
            triple = kinds[i % 3](n, rng)
        triples.append(triple)
    return triples


@dataclass(frozen=True)
class Case:
    """One case of a property: a triple, the sum's digit checked (None: the whole sum), a stratum."""

    triple: Triple
    column: int | None = None
    stratum: int | None = None


Verifier = Callable[[Sequence[Case], Adder, int], list[bool]]


@dataclass(frozen=True)
class Property:
    id: str
    statement: str
    generate: Callable[[int, Random], list[Case]]
    verify: Verifier
    coverage: str


@dataclass
class PropertyResult:
    id: str
    cases: int
    passed: int
    # stratum: (passed, cases)
    strata: dict[int, tuple[int, int]] = field(default_factory=dict[int, tuple[int, int]])
    failures: list[Case] = field(default_factory=list[Case])  # shrunk, the first FAILURES_KEPT

    @property
    def holds(self) -> bool:
        return self.passed == self.cases

    @property
    def accuracy(self) -> float:
        return self.passed / self.cases

    def reach(self) -> int:
        """M3's L*: the largest distance L with every case at 1..L right (0 when L = 1 fails)."""
        reach = 0
        for stratum in sorted(self.strata):
            passed, cases = self.strata[stratum]
            if passed < cases:
                break
            reach = stratum
        return reach


# verifiers


def oracle(cases: Sequence[Case], f: Adder, n: int) -> list[bool]:
    """Each case's answer is the sum: the whole of it, or the digit at the case's column."""
    answers = f([case.triple for case in cases])
    return [
        answer == sum(case.triple)
        if case.column is None
        else sum_digit(answer, case.column) == sum_digit(sum(case.triple), case.column)
        for case, answer in zip(cases, answers, strict=True)
    ]


def _orders(triple: Triple) -> list[Triple]:
    return [(x, y, z) for x, y, z in itertools.permutations(triple)]


def commutes(cases: Sequence[Case], f: Adder, n: int) -> list[bool]:
    """A1: the 6 orders of each triple give one answer."""
    answers = f([order for case in cases for order in _orders(case.triple)])
    return [len(set(answers[6 * i : 6 * i + 6])) == 1 for i in range(len(cases))]


def associates(cases: Sequence[Case], f: Adder, n: int) -> list[bool]:
    """
    A2: f(f(a, b, 0), c, 0) = f(a, f(b, c, 0), 0) = f(a, b, c), operands of n - 1 digits; a partial
    sum that doesn't fit n digits fails.
    """
    first = f([q for case in cases for q in _first_stage(case.triple)])
    partials = [(first[3 * i], first[3 * i + 1], first[3 * i + 2]) for i in range(len(cases))]
    fits = [ab < BASE**n and bc < BASE**n for ab, bc, _abc in partials]
    second_queries = [
        q
        for case, (ab, bc, _abc), ok in zip(cases, partials, fits, strict=True)
        if ok
        for q in ((ab, case.triple[2], 0), (case.triple[0], bc, 0))
    ]
    second = iter(f(second_queries)) if second_queries else iter(())
    holds: list[bool] = []
    for (_ab, _bc, abc), ok in zip(partials, fits, strict=True):
        if not ok:
            holds.append(False)
            continue
        left, right = next(second), next(second)
        holds.append(left == right == abc)
    return holds


def _first_stage(triple: Triple) -> tuple[Triple, Triple, Triple]:
    a, b, c = triple
    return ((a, b, 0), (b, c, 0), (a, b, c))


# generators


def _all_triples(n: int) -> Iterator[Triple]:
    r = range(BASE**n)
    return ((a, b, c) for a in r for b in r for c in r)


def _adds_three(n: int, rng: Random) -> list[Case]:
    if BASE ** (3 * n) <= EXHAUSTIVE_LIMIT:
        return [Case(t) for t in _all_triples(n)]
    return [Case(uniform_triple(n, rng)) for _ in range(SAMPLED)]


def _adds_two(n: int, rng: Random) -> list[Case]:
    if 3 * BASE ** (2 * n) <= EXHAUSTIVE_LIMIT:
        r = range(BASE**n)
        return [Case(t) for x in r for y in r for t in ((0, x, y), (x, 0, y), (x, y, 0))]
    return [Case(two_operand_triple(n, rng)) for _ in range(SAMPLED)]


def _held_out(n: int, rng: Random) -> list[Case]:
    if BASE ** (3 * n) <= EXHAUSTIVE_LIMIT:
        return [Case(t) for t in _all_triples(n) if held_out(t, n)]
    cases: list[Case] = []
    while len(cases) < SAMPLED:
        triple = uniform_triple(n, rng)
        if held_out(triple, n):
            cases.append(Case(triple))
    return cases


def _commutativity(n: int, rng: Random) -> list[Case]:
    cases = [Case(uniform_triple(n, rng)) for _ in range(RELATION_SAMPLED // 2)]
    for _ in range(RELATION_SAMPLED - len(cases)):
        x, y = rng.randrange(BASE**n), rng.randrange(BASE**n)
        triple = [x, x, y]
        rng.shuffle(triple)
        cases.append(Case((triple[0], triple[1], triple[2])))
    return cases


def _associativity(n: int, rng: Random) -> list[Case]:
    return [Case(uniform_triple(n - 1, rng)) for _ in range(RELATION_SAMPLED)]


def _identity(n: int, rng: Random) -> list[Case]:
    return [Case(t) for a in range(BASE**n) for t in ((a, 0, 0), (0, a, 0), (0, 0, a))]


def _column_sum(n: int, rng: Random) -> list[Case]:
    return [Case(triple_of_sums(sums, rng)) for sums in itertools.product(range(BASE), repeat=n) for _ in range(4)]


def _carry_from_reset(n: int, rng: Random) -> list[Case]:
    # a reset at column j, every value; the digit checked is j + 1's
    return [
        Case(triple_of_sums(sums, rng), column=j + 1, stratum=reset // BASE)
        for j in range(n)
        for reset in RESETS
        for _ in range(16)
        for sums in [[reset if t == j else rng.choice(COLUMN_SUMS) for t in range(n)]]
    ]


def _carry_over_distance(n: int, rng: Random) -> list[Case]:
    # for each distance L in 1..n: the checked column t, a reset at t - 1 - L (column 0's carry in
    # when that is -1), L dependent columns between, every other column uniform
    cases: list[Case] = []
    for length in range(1, n + 1):
        for _ in range(PER_STRATUM):
            t = rng.randrange(length, n + 1)
            r = t - 1 - length
            sums = [
                rng.choice(RESETS) if i == r else rng.choice(DEPENDENT) if r < i < t else rng.choice(COLUMN_SUMS)
                for i in range(n)
            ]
            cases.append(Case(triple_of_sums(sums, rng), column=t, stratum=length))
    return cases


def _final_carry(n: int, rng: Random) -> list[Case]:
    cases: list[Case] = []
    for value in range(BASE):
        while sum(case.stratum == value for case in cases) < PER_STRATUM:
            triple = pattern_triple(n, rng)
            if carries(column_sums(triple, n))[n] == value:
                cases.append(Case(triple, column=n, stratum=value))
    return cases


def _every_pattern(n: int, rng: Random) -> list[Case]:
    per_pattern = max(1, PATTERN_CASES // len(COLUMN_SUMS) ** n)
    return [
        Case(triple_of_sums(sums, rng)) for sums in itertools.product(COLUMN_SUMS, repeat=n) for _ in range(per_pattern)
    ]


CATALOGUE: tuple[Property, ...] = (
    Property("B1", "adds three: f(a, b, c) = a + b + c", _adds_three, oracle, "every triple, or 100,000 sampled"),
    Property(
        "B2", "adds two: a zero in any one slot", _adds_two, oracle, "every pair in every slot, or 100,000 sampled"
    ),
    Property("B3", "adds three on the held-out tenth (D8)", _held_out, oracle, "the whole tenth, or 100,000 sampled"),
    Property("A1", "commutativity: the 6 orders give one answer", _commutativity, commutes, "10,000 triples × 6"),
    Property(
        "A2",
        "associativity: f(f(a, b, 0), c, 0) = f(a, f(b, c, 0), 0) = f(a, b, c)",
        _associativity,
        associates,
        "10,000 triples of n - 1 digits",
    ),
    Property("A3", "identity: f(a, 0, 0) = f(0, a, 0) = f(0, 0, a) = a", _identity, oracle, "every a, every slot"),
    Property("M1", "column sum: with no carries, digit t is s_t", _column_sum, oracle, "every pattern over 0..2, × 4"),
    Property(
        "M2", "carry from a reset: the next column's digit", _carry_from_reset, oracle, "every column × reset × 16"
    ),
    Property(
        "M3",
        "carry over distance L: the digit at distance L above a reset",
        _carry_over_distance,
        oracle,
        "1,000 per L in 1..n",
    ),
    Property("M4", "final carry: the sum's top digit, 0, 1 or 2", _final_carry, oracle, "1,000 per value"),
    Property(
        "M5",
        "every carry pattern: every one of the 7^n",
        _every_pattern,
        oracle,
        "every pattern × max(1, 10,000 // 7^n)",
    ),
)


def property_cases(prop: Property, n: int) -> list[Case]:
    """A property's cases at n: the same for every candidate and every run."""
    return prop.generate(n, Random(f"addition-{prop.id}-{n}"))


def evaluate(
    f: Adder, n: int, properties: Sequence[Property] = CATALOGUE, limit: int | None = None
) -> list[PropertyResult]:
    """
    Each property verified against f, its failures shrunk. limit takes every k-th case (k the
    smallest that leaves at most limit): a fast screen of the same cases.
    """
    results: list[PropertyResult] = []
    for prop in properties:
        cases = property_cases(prop, n)
        if limit is not None and len(cases) > limit:
            cases = cases[:: -(-len(cases) // limit)]
        holds = prop.verify(cases, f, n)
        result = PropertyResult(prop.id, len(cases), sum(holds))
        for case, ok in zip(cases, holds, strict=True):
            if case.stratum is not None:
                passed, count = result.strata.get(case.stratum, (0, 0))
                result.strata[case.stratum] = (passed + ok, count + 1)
            if not ok and len(result.failures) < FAILURES_KEPT:
                result.failures.append(shrink(case, prop.verify, f, n))
        results.append(result)
    return results


def shrink(case: Case, verify: Verifier, f: Adder, n: int) -> Case:
    """
    A failing case made minimal: whole columns zeroed from the top while it still fails, then
    single digits lowered; the column checked and the stratum kept.
    """

    def fails(triple: Triple) -> bool:
        return not verify([Case(triple, case.column, case.stratum)], f, n)[0]

    operands: list[list[int]] = [list(digits(x, n)) for x in case.triple]
    changed = True
    while changed:
        changed = False
        for t in reversed(range(n)):
            if any(o[t] for o in operands):
                trial = [[0 if i == t else d for i, d in enumerate(o)] for o in operands]
                if fails(_triple(trial)):
                    operands, changed = trial, True
        for o_index in range(3):
            for t in range(n):
                while operands[o_index][t] > 0:
                    trial = [list(o) for o in operands]
                    trial[o_index][t] -= 1
                    if not fails(_triple(trial)):
                        break
                    operands, changed = trial, True
    return Case(_triple(operands), case.column, case.stratum)


def _triple(operands: Sequence[Sequence[int]]) -> Triple:
    return (from_digits(operands[0]), from_digits(operands[1]), from_digits(operands[2]))
