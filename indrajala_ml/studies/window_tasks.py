"""
The window-recurrent study's tasks (docs/window-recurrent-claims-workplan.md, stage 5, D14-D18):
what a window-recurrent network learns, as the harness (window_recurrent_study) and the
certificate (window_recurrent_certificate) read it.

A task is a target Mealy machine over column tokens (its start, its transition and its output for
each state and token), an encoding of its cases as column tokens in its reading order, a training
mixture and a catalogue of tests. Its labels are the machine's outputs: an example's tokens (the
alignment's zero columns, the case's columns and zero columns through the final window) run
through the machine give one label a column, so the certificate checks a network against the very
machine its labels come from. Zero columns leave each task's start state where it is (no carry; A
and B both 0), so an alignment's zero columns are part of the word like any other.

- **addition:** three base-3 operands, least significant first, a column's token 9 a_t + 3 b_t +
  c_t (27); the carry automaton, 3 states, its output the sum's digit (3 classes); addition_data's
  catalogue.
- **modular-division-p:** two base-3 numbers, most significant first, a column's token 3 a_i + b_i
  (9); the modular automaton, p^2 states (A mod p, B mod p), its output the running quotient or
  undefined (p + 1 classes); modular_data's catalogue.
- **modular-sum-q:** the same pairs and tokens; the automaton of (A + B) mod q, q states, its output
  the running sum (q classes); modular_data's sum catalogue (the claims workplan's E8).
"""

from __future__ import annotations

import functools
import itertools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from indrajala_ml.data import addition_data as ad
from indrajala_ml.data import modular_data as md
from indrajala_ml.data.addition_data import BASE

IntArray = npt.NDArray[np.int64]
Labels = tuple[int, ...]
# a batch of cases to each one's labels, from its first column (past the alignment) to its end
Predict = Callable[[Sequence[Any]], list[Labels]]


@dataclass(frozen=True)
class Coordinates:
    """
    A machine's states named by numbers mod modulus (the factored-state workplan, D9): values[s] the
    coordinates of state s, one per name (division's A and B), so the certificate can name what a
    factor of a network's state holds, as a linear form of them.
    """

    names: tuple[str, ...]
    values: IntArray  # (S, len(names))
    modulus: int


@dataclass(frozen=True)
class Mealy:
    """
    A Mealy machine over tokens 0..T-1: transition[s, t] the next state, output[s, t] the label;
    its states' coordinates, if named.
    """

    transition: IntArray  # (S, T)
    output: IntArray  # (S, T)
    start: int = 0
    coordinates: Coordinates | None = None

    @property
    def states(self) -> int:
        return int(self.transition.shape[0])

    @property
    def tokens(self) -> int:
        return int(self.transition.shape[1])

    @property
    def classes(self) -> int:
        return int(self.output.max()) + 1

    @functools.cached_property
    def _table(self) -> list[list[tuple[int, int]]]:
        return [
            [(int(r), int(o)) for r, o in zip(row_t, row_o, strict=True)]
            for row_t, row_o in zip(self.transition, self.output, strict=True)
        ]

    def run(self, tokens: Sequence[int]) -> Labels:
        """The labels of a word of tokens from the start."""
        table, state = self._table, self.start
        labels: list[int] = []
        for token in tokens:
            state, label = table[state][token]
            labels.append(label)
        return tuple(labels)

    def windows(self, windows: IntArray) -> tuple[IntArray, IntArray]:
        """From each state (axis 0): each window's labels (S, N, k) and the state after it (S, N)."""
        labels = np.zeros((self.states, *windows.shape), dtype=np.int64)
        after = np.zeros((self.states, windows.shape[0]), dtype=np.int64)
        for s in range(self.states):
            state = np.full(windows.shape[0], s, dtype=np.int64)
            for j in range(windows.shape[1]):
                labels[s, :, j] = self.output[state, windows[:, j]]
                state = self.transition[state, windows[:, j]]
            after[s] = state
        return labels, after

    def distinguishing_length(self) -> int | None:
        """
        The least k such that every pair of states is told apart by some word of k tokens (Chow 1978's
        k-distinguishability: the outputs differ somewhere along it), or None if some pair never is.
        """
        classes, count = [0] * self.states, 1
        for k in itertools.count(1):
            keys = [
                (
                    classes[s],
                    *((int(self.output[s, t]), classes[int(self.transition[s, t])]) for t in range(self.tokens)),
                )
                for s in range(self.states)
            ]
            index = {key: i for i, key in enumerate(dict.fromkeys(keys))}
            classes = [index[key] for key in keys]
            if len(index) == self.states:
                return k
            if len(index) == count:
                return None
            count = len(index)
        raise AssertionError("unreachable")

    def is_congruence(self, labels: IntArray | Sequence[int]) -> bool:
        """
        Whether the partition of the states by `labels` is respected by every transition: states with
        one label go, on every token, to states with one label. Then the labels' machine is a
        homomorphic image of this one (an approximation, in Egri-Nagy and Nehaniv's terms).
        """
        for t in range(self.tokens):
            image: dict[int, int] = {}
            for s in range(self.states):
                after = int(labels[int(self.transition[s, t])])
                if image.setdefault(int(labels[s]), after) != after:
                    return False
        return True

    def distinguishing(self, s: int, r: int) -> list[int]:
        """The shortest word (first in token order) whose last label differs from s and from r."""
        assert s != r
        parent: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {(s, r): None}
        queue = [(s, r)]
        for pair in queue:
            for t in range(self.tokens):
                if self.output[pair[0], t] != self.output[pair[1], t]:
                    word = [t]
                    at: tuple[int, int] = pair
                    while (step := parent[at]) is not None:
                        at, token = step
                        word.append(token)
                    return word[::-1]
                following = (int(self.transition[pair[0], t]), int(self.transition[pair[1], t]))
                if following not in parent:
                    parent[following] = (pair, t)
                    queue.append(following)
        raise ValueError(f"states {s} and {r} are equivalent: the machine isn't minimal")


def carry_machine() -> Mealy:
    """Base-3 addition of three operands: states the carry 0..2, a column's token 9 a + 3 b + c."""
    sums = np.array([t // (BASE * BASE) + t // BASE % BASE + t % BASE for t in range(BASE**3)])
    totals = np.arange(3)[:, np.newaxis] + sums[np.newaxis, :]
    return Mealy(totals // BASE, totals % BASE, coordinates=Coordinates(("carry",), np.arange(3)[:, np.newaxis], 3))


@functools.cache
def modular_machine(p: int) -> Mealy:
    """Modular division: state A * p + B (each mod p), a column's token 3 a + b, the output the running quotient."""
    transition = np.zeros((p * p, BASE * BASE), dtype=np.int64)
    output = np.zeros((p * p, BASE * BASE), dtype=np.int64)
    for A in range(p):
        for B in range(p):
            for t in range(BASE * BASE):
                A2, B2 = (BASE * A + t // BASE) % p, (BASE * B + t % BASE) % p
                transition[A * p + B, t] = A2 * p + B2
                output[A * p + B, t] = md.quotient(A2, B2, p)
    coordinates = np.array([[A, B] for A in range(p) for B in range(p)], dtype=np.int64)
    return Mealy(transition, output, coordinates=Coordinates(("A", "B"), coordinates, p))


@functools.cache
def modular_sum_machine(q: int) -> Mealy:
    """Modular sum: state (A + B) mod q, a column's token 3 a + b, the output the running sum."""
    transition = np.zeros((q, BASE * BASE), dtype=np.int64)
    for total in range(q):
        for t in range(BASE * BASE):
            transition[total, t] = (BASE * total + t // BASE + t % BASE) % q
    return Mealy(transition, transition.copy(), coordinates=Coordinates(("total",), np.arange(q)[:, np.newaxis], q))


CARRY = carry_machine()


class Outcome(Protocol):
    """A property's result, as the harness reads it."""

    id: str
    cases: int
    passed: int

    @property
    def holds(self) -> bool: ...


class Task(Protocol):
    """What the harness and the certificate need of a task (the module's docstring)."""

    @property
    def name(self) -> str: ...

    @property
    def machine(self) -> Mealy: ...

    @property
    def held_out_id(self) -> str:
        """The held-out tenth's property: checked from the harness's HOLD_OUT_FROM, never tested."""
        ...

    @property
    def codes_id(self) -> str:
        """The property whose first cases count the codes used."""
        ...

    @property
    def reach_id(self) -> str | None:
        """The stratified property whose L* the test pass records (None: the task has none)."""
        ...

    def column_tokens(self, case: Any, n: int) -> tuple[int, ...]: ...

    def training_case(self, kind: int, n: int, rng: Random, hold_out: bool) -> Any: ...

    def hash(self, case: Any, n: int) -> int: ...

    def catalogue(
        self, sampled: int, relation_sampled: int, per_stratum: int, doubling: bool = False
    ) -> tuple[Any, ...]: ...

    def property_cases(self, prop: Any, n: int) -> list[Any]: ...

    def evaluate(self, predict: Predict, n: int, properties: Sequence[Any], limit: int | None = None) -> list[Any]: ...

    def reach(self, result: Any) -> int: ...

    def result_json(self, result: Any) -> dict[str, Any]: ...

    def case_json(self, columns: Sequence[int]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Addition:
    """Three-operand base-3 addition, least significant first (the matrix's task)."""

    name: str = "addition"
    machine: Mealy = CARRY
    held_out_id: str = "B3"
    codes_id: str = "B1"
    reach_id: str = "M3"

    def column_tokens(self, case: ad.Triple, n: int) -> tuple[int, ...]:
        a, b, c = (ad.digits(x, n) for x in case)
        return tuple(BASE * BASE * a[t] + BASE * b[t] + c[t] for t in range(n))

    def training_case(self, kind: int, n: int, rng: Random, hold_out: bool) -> ad.Triple:
        return ad.training_triple(kind, n, rng, hold_out=hold_out)

    def hash(self, case: ad.Triple, n: int) -> int:
        return ad.triple_hash(case, n)

    def catalogue(
        self, sampled: int, relation_sampled: int, per_stratum: int, doubling: bool = False
    ) -> tuple[ad.Property, ...]:
        return ad.catalogue(sampled, relation_sampled, per_stratum, doubling)

    def property_cases(self, prop: ad.Property, n: int) -> list[ad.Triple]:
        return [case.triple for case in ad.property_cases(prop, n)]

    def evaluate(
        self, predict: Predict, n: int, properties: Sequence[ad.Property], limit: int | None = None
    ) -> list[ad.PropertyResult]:
        def add(triples: Sequence[ad.Triple]) -> list[int]:
            return [ad.from_digits(labels) for labels in predict(triples)]

        return ad.evaluate(add, n, properties, limit=limit, shrink=False)

    def reach(self, result: ad.PropertyResult) -> int:
        return result.reach()

    def result_json(self, result: ad.PropertyResult) -> dict[str, Any]:
        return {
            "cases": result.cases,
            "passed": result.passed,
            "strata": {str(k): list(v) for k, v in sorted(result.strata.items())},
            "failures": [{"triple": list(c.triple), "column": c.column} for c in result.failures],
        }

    def case_json(self, columns: Sequence[int]) -> dict[str, Any]:
        """The columns as an addition at alignment 0, least significant first: its triple and n."""
        operands = [0, 0, 0]
        for t, token in enumerate(columns):
            for i, digit in enumerate((token // (BASE * BASE), token // BASE % BASE, token % BASE)):
                operands[i] += digit * BASE**t
        return {"triple": operands, "n": len(columns)}


@dataclass(frozen=True)
class ModularDivision:
    """(a / b) mod p of two base-3 numbers, most significant first, the running quotient at every column."""

    p: int

    @property
    def name(self) -> str:
        return f"modular-division-{self.p}"

    @property
    def machine(self) -> Mealy:
        return modular_machine(self.p)

    held_out_id: str = "U2"
    codes_id: str = "U1"
    reach_id: str | None = "S2"

    def column_tokens(self, case: md.Pair, n: int) -> tuple[int, ...]:
        return md.column_tokens(case, n)

    def training_case(self, kind: int, n: int, rng: Random, hold_out: bool) -> md.Pair:
        return md.training_pair(kind, n, rng, hold_out=hold_out)

    def hash(self, case: md.Pair, n: int) -> int:
        return md.pair_hash(case, n)

    def catalogue(
        self, sampled: int, relation_sampled: int, per_stratum: int, doubling: bool = False
    ) -> tuple[md.Property, ...]:
        return md.catalogue(self.p, sampled, per_stratum, doubling)

    def property_cases(self, prop: md.Property, n: int) -> list[md.Pair]:
        return [case.pair for case in md.property_cases(prop, n)]

    def evaluate(
        self, predict: Predict, n: int, properties: Sequence[md.Property], limit: int | None = None
    ) -> list[md.PropertyResult]:
        def divide(pairs: Sequence[md.Pair]) -> list[md.Answers]:
            return [labels[:n] for labels in predict(pairs)]

        return md.evaluate(divide, n, properties, limit=limit)

    def reach(self, result: md.PropertyResult) -> int:
        return result.reach()

    def result_json(self, result: md.PropertyResult) -> dict[str, Any]:
        return {
            "cases": result.cases,
            "passed": result.passed,
            "strata": {str(k): list(v) for k, v in sorted(result.strata.items())},
            "failures": [{"pair": list(c.pair)} for c in result.failures],
        }

    def case_json(self, columns: Sequence[int]) -> dict[str, Any]:
        """The columns as a pair at alignment 0, most significant first: the pair and n."""
        a = b = 0
        for token in columns:
            a, b = BASE * a + token // BASE, BASE * b + token % BASE
        return {"pair": [a, b], "n": len(columns)}


@dataclass(frozen=True)
class ModularSum(ModularDivision):
    """(a + b) mod q of two base-3 numbers, most significant first, the running sum at every column (E8)."""

    reach_id: str | None = None  # no undefined class, so no S2

    @property
    def name(self) -> str:
        return f"modular-sum-{self.p}"

    @property
    def machine(self) -> Mealy:
        return modular_sum_machine(self.p)

    def catalogue(
        self, sampled: int, relation_sampled: int, per_stratum: int, doubling: bool = False
    ) -> tuple[md.Property, ...]:
        return md.sum_catalogue(self.p, sampled, per_stratum)


ADDITION = Addition()


def task_named(name: str) -> Task:
    """addition, modular-division-p for p in md.PRIMES, or modular-sum-q for q >= 2."""
    if name == ADDITION.name:
        return ADDITION
    if found := re.fullmatch(r"modular-sum-(\d+)", name):
        assert int(found[1]) >= 2, f"a modular sum's q is at least 2; got {found[1]}"
        return ModularSum(int(found[1]))
    found = re.fullmatch(r"modular-division-(\d+)", name)
    assert found is not None and int(found[1]) in md.PRIMES, (
        f"a task is addition, modular-division-p (p in {md.PRIMES}) or modular-sum-q"
    )
    return ModularDivision(int(found[1]))
