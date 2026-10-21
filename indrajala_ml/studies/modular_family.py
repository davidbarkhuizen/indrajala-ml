"""
The modular family (the modular-family workplan): one window-recurrent network answering
several modular operations on the same columns at every window, so a factor of its state gets a
dense signal for a residue (the sum's and difference's counters) where division alone gives only
its zero tests.

A column is the token `3 a + b` of two ternary digits, most significant first (modular_data's
encoding); after each column the prefixes read so far are `A` and `B`. A window is the column
followed by one query token per extra operation (D1): the column's own output is the running
quotient `A / B mod p` (or undefined, class `p`), each query's output its operation's answer:

- sum: `(A + B) mod p`;
- difference: `(A - B) mod p`.

`QUOTIENT` alone (no queries) is division as the window studies train it, the probe's control;
`SUM` the family without the difference (one dense counter).
The probe trains one network by the curriculum (step `j` trains lengths `1..j`) within a
wall-clock budget, diagnosing it as it goes: each operation's accuracy, the quotient's by class,
and each factor's share of `A`, `B`, the sum, the difference and the zero tests
(window_diagnostics.share).

The probe runs any target the same way (the general algebras, docs/general-algebras-workplan.md,
D6): a target is an encoding (`Target`), division's `Encoding` one, `MachineEncoding` (any
window_tasks machine, its own task, one token a window) another. The fork, the trigger, the guard
and the readouts read only what a Target gives: its windows and answers, each answer's kind (D7:
a machine's by its token), which kinds are the certified answer's (the quotient's for division,
every kind for a machine), its task and the quantities its diagnosis shares.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import random
import re
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from indrajala_ml.data import addition_data as ad
from indrajala_ml.data import modular_data as md
from indrajala_ml.model.layers.array.branch_layers import BranchArrayLayer
from indrajala_ml.model.networks.window_recurrent_array_network import WindowRecurrentRustArrayNetwork
from indrajala_ml.model.persistence.format2 import network_to_json
from indrajala_ml.model.persistence.load_network import load_network, network_from_state
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import (
    BEFORE_STATE,
    OUTPUT,
    AttentionPooling,
    Codes,
    Factor,
    WindowRecurrence,
)
from indrajala_ml.studies import certificate_guard as cg
from indrajala_ml.studies import inference
from indrajala_ml.studies import machine_reading as mr
from indrajala_ml.studies import window_diagnostics as wd
from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_recurrent_study as wr
from indrajala_ml.studies import window_tasks as wt
from indrajala_ml.studies.collapse import Trigger

BASE = md.BASE
COLUMNS = BASE * BASE  # a column's tokens, 3 a + b

# the operations a query token asks for, after the column's own quotient
OPERATIONS: dict[str, Callable[[int, int, int], int]] = {
    "sum": lambda A, B, p: (A + B) % p,
    "difference": lambda A, B, p: (A - B) % p,
    "product": lambda A, B, p: (A * B) % p,  # not an approximation (the approximation claim)
}
FAMILY = ("sum", "difference")
SUM = ("sum",)  # the family without the difference (the division workplan's stage 3 option)
PRODUCT = ("product",)
QUOTIENT: tuple[str, ...] = ()
VOCABULARY = ("sum", "difference")  # the operations with a token: every study's before the approximation claim
WIDE = ("sum", "difference", "product")  # the approximation claim's, the same for its every arm

Example = wr.Example

LONG_PROPERTIES = ("U1", "S1")  # the long tests' (the division workplan's D4)


@dataclass(frozen=True)
class Encoding:
    """
    A window's layout: the column, then a query token per operation, each its own answer. The
    vocabulary is the operations with a token (the embedding's rows), so arms compared with each
    other share it whatever they query; the default is every study's before the product's.
    """

    p: int
    queries: tuple[str, ...]
    vocabulary: tuple[str, ...] = VOCABULARY

    def __post_init__(self) -> None:
        assert set(self.queries) <= set(self.vocabulary) <= set(OPERATIONS), (
            f"queries {self.queries} need tokens in the vocabulary {self.vocabulary}, of {tuple(OPERATIONS)}"
        )

    @property
    def width(self) -> int:
        return 1 + len(self.queries)

    @property
    def tokens(self) -> int:
        return COLUMNS + len(self.vocabulary)

    @property
    def classes(self) -> int:
        return self.p + 1  # residues and undefined

    def query_token(self, operation: str) -> int:
        return COLUMNS + self.vocabulary.index(operation)

    @property
    def query_tokens(self) -> tuple[int, ...]:
        return tuple(self.query_token(operation) for operation in self.queries)

    @property
    def task(self) -> wt.Task:
        return wt.task_named(f"modular-division-{self.p}")

    @property
    def target_states(self) -> int:
        return self.p * self.p

    def draw(self, n: int, rng: random.Random) -> list[int]:
        return columns(n, rng)

    def kinds(self, tokens: Sequence[float], labels: Sequence[int]) -> list[str]:
        """Each answer's kind: the quotient by class (0, undefined, nonzero), each query by its operation."""
        names: list[str] = []
        for i, label in enumerate(labels):
            slot = i % self.width
            if slot == 0:
                names.append(
                    "quotient-0" if label == 0 else "quotient-undefined" if label == self.p else "quotient-nonzero"
                )
            else:
                names.append(self.queries[slot - 1])
        return names

    def certified(self, kind: str) -> bool:
        """Whether a kind is the quotient's, the answer certified."""
        return kind.startswith("quotient")

    @property
    def quantities(self) -> dict[str, Callable[[Any], int]]:
        """What the diagnosis shares each factor with, of a state (A, B)."""
        p = self.p
        return {
            "A": lambda s: s[0],
            "B": lambda s: s[1],
            "S": lambda s: (s[0] + s[1]) % p,
            "D": lambda s: (s[0] - s[1]) % p,
            "A=0": lambda s: int(s[0] == 0),
            "B=0": lambda s: int(s[1] == 0),
        }

    def states(self, columns: Sequence[int]) -> list[tuple[int, int]]:
        """(A, B) mod p after each column."""
        a = b = 0  # the prefixes mod p
        states: list[tuple[int, int]] = []
        for token in columns:
            a, b = (BASE * a + token // BASE) % self.p, (BASE * b + token % BASE) % self.p
            states.append((a, b))
        return states

    def example(self, columns: Sequence[int]) -> Example:
        """The windows' tokens and each token's answer (states' tables: as `states` and the answers per column)."""
        tokens: list[float] = []
        labels: list[int] = []
        windows, answers, successors = self._windows, self._answers, self._successors
        state = 0  # (A, B) mod p as A p + B
        for token in columns:
            state = successors[state][token]
            tokens += windows[token]
            labels += answers[state]
        return tuple(tokens), tuple(labels)

    @functools.cached_property
    def _windows(self) -> tuple[tuple[float, ...], ...]:
        """Each column token's window: the token, then a token per query."""
        queries = tuple(float(self.query_token(operation)) for operation in self.queries)
        return tuple((float(token), *queries) for token in range(COLUMNS))

    @functools.cached_property
    def _answers(self) -> tuple[tuple[int, ...], ...]:
        """Each state's answers, A p + B: the quotient's, then each query's."""
        p = self.p
        return tuple(
            (md.quotient(A, B, p), *(OPERATIONS[operation](A, B, p) for operation in self.queries))
            for A in range(p)
            for B in range(p)
        )

    @functools.cached_property
    def _successors(self) -> tuple[tuple[int, ...], ...]:
        """Each state's next, A p + B, after each column token (as `states`)."""
        p = self.p
        return tuple(
            tuple((BASE * A + token // BASE) % p * p + (BASE * B + token % BASE) % p for token in range(COLUMNS))
            for A in range(p)
            for B in range(p)
        )


@dataclass(frozen=True)
class MachineEncoding:
    """
    Any machine's own task as the probe's target (D6): a window is one token, its answer the
    machine's output, its kind its token (D7), every kind the certified answer's; the diagnosis
    shares each factor with each coordinate of the state (window_diagnostics.coordinates).
    """

    task: wt.Task
    width: int = 1
    query_tokens: tuple[int, ...] = ()

    @property
    def tokens(self) -> int:
        return self.task.machine.tokens

    @property
    def classes(self) -> int:
        return self.task.machine.classes

    @property
    def target_states(self) -> int:
        return self.task.machine.states

    def draw(self, n: int, rng: random.Random) -> list[int]:
        """n uniform tokens, then the identity (token 0), as division's final zero column: its answer
        needs the state the word left, so even one token trains the state (a probe without it,
        2026-10-17: 5 of 6 runs at 6 states kept one code from step 1, where no answer needs the state)."""
        return [rng.randrange(self.tokens) for _ in range(n)] + [0]

    def example(self, columns: Sequence[int]) -> Example:
        return tuple(float(token) for token in columns), self.task.machine.run(columns)

    def kinds(self, tokens: Sequence[float], labels: Sequence[int]) -> list[str]:
        return [f"token-{int(token)}" for token in tokens]

    def certified(self, kind: str) -> bool:
        return True

    @property
    def quantities(self) -> dict[str, Callable[[Any], int]]:
        return dict(wd.coordinates(self.task.machine))

    def states(self, columns: Sequence[int]) -> list[int]:
        return self.task.machine.path(columns)


class Target(Protocol):
    """What the probe needs of a target (D6): Encoding's and MachineEncoding's members, documented there."""

    @property
    def width(self) -> int: ...

    @property
    def tokens(self) -> int: ...

    @property
    def classes(self) -> int: ...

    @property
    def query_tokens(self) -> tuple[int, ...]: ...

    @property
    def task(self) -> wt.Task: ...

    @property
    def target_states(self) -> int: ...

    @property
    def quantities(self) -> Mapping[str, Callable[[Any], int]]: ...

    def draw(self, n: int, rng: random.Random) -> list[int]: ...

    def example(self, columns: Sequence[int]) -> Example: ...

    def kinds(self, tokens: Sequence[float], labels: Sequence[int]) -> list[str]: ...

    def certified(self, kind: str) -> bool: ...

    def states(self, columns: Sequence[int]) -> Sequence[Hashable]: ...


def target_named(name: str, queries: tuple[str, ...] = (), vocabulary: tuple[str, ...] = VOCABULARY) -> Target:
    """modular-division-p's Encoding (with queries and vocabulary), or any other task's MachineEncoding."""
    if found := re.fullmatch(r"modular-division-(\d+)", name):
        return Encoding(int(found[1]), queries, vocabulary)
    return MachineEncoding(wt.task_named(name))


def columns(n: int, rng: random.Random) -> list[int]:
    """n uniform columns, then the final zero column (the window studies' final zero window, at
    k 1: its answer needs the state the data left, so even one column trains the state)."""
    return [rng.randrange(COLUMNS) for _ in range(n)] + [0]


def network(
    encoding: Target,
    factors: Sequence[tuple[int, int]],
    seed: int,
    reads: str = BEFORE_STATE,
    width: int = wr.D,
    branches: bool = False,
    answer_branch: bool = False,
) -> Any:
    """The transformer window (W11) over the encoding's tokens, a factored state (D3), Adam; the
    poolings reading before the first StateInput (D10) or, with reads OUTPUT, the window's output.
    With branches, the window branches workplan's window (wr.branched_window), each factor's
    pooling reading its own branch's end (reads OUTPUT); with answer_branch, the one-code-one-state
    workplan's R1 (wr.answer_branch_window), every factor reading the shared state branch."""
    assert not (branches or answer_branch) or reads == OUTPUT, (
        "with branches each factor reads a branch's end: reads OUTPUT"
    )
    assert not (branches and answer_branch), "the window branches or the answers' branch, not both"
    recurrence = WindowRecurrence(
        tuple(Factor(s, Codes(m)) for s, m in factors), False, pooling=AttentionPooling(reads)
    )
    if branches:
        specs = wr.branched_window(encoding.tokens, encoding.classes, len(factors), width)
    elif answer_branch:
        specs = wr.answer_branch_window(encoding.tokens, encoding.classes, len(factors), width)
    else:
        specs = wr.transformer_window(encoding.tokens, encoding.classes, width)
    return WindowRecurrentRustArrayNetwork.randomized((encoding.width,), specs, recurrence, Adam(), seed=seed)


@dataclass(frozen=True)
class Settings:
    """The probe's training (D4)."""

    epoch_examples: int = 2**15
    batch_size: int = 64
    rate: float = 0.003
    patience: int = 10
    max_epochs: int = 60
    screen: int = 200  # cases a length, each epoch's screen
    diagnose_every: int = 5
    lengths: tuple[int, ...] = (3, 6, 12)  # the diagnosis's
    cases: int = 200  # a diagnosis length's
    long: tuple[int, ...] = (24, 48)  # the quotient's, at a step's end


PROBE = Settings()


@dataclass(frozen=True)
class Fork:
    """
    Snapshot and fork on collapse (the collapse-fork workplan, retired:
    `git show 89cf3a3:docs/collapse-fork-workplan.md`): within a step, the network at the step's
    best value of the trigger's signal is kept; an epoch the trigger calls a collapse
    (collapse.Trigger; by default the screen below 0.6 of the step's peak, the workplan's D1) sends
    the run back to the snapshot, cools the step's rate by `cool` and takes its data from a new
    stream; at most `limit` forks a step.
    """

    trigger: Trigger = field(default_factory=Trigger)
    cool: float = 0.5  # D3
    limit: int = 4  # D4


def screen(net: Any, encoding: Target, screens: Mapping[int, Sequence[Example]]) -> tuple[float, dict[str, float]]:
    """
    A step's screen: the share of every answer right over its screens (its training lengths), and
    each answer kind's share over them (the p11-trigger workplan's D4: a trigger's signals).
    """
    tallies = [accuracy(net, encoding, rows) for rows in screens.values()]
    correct = sum(c for t in tallies for c, _ in t.values())
    pooled = correct / sum(a for t in tallies for _, a in t.values())
    kinds: dict[str, list[int]] = {}
    for tally in tallies:
        for name, (right, all_) in tally.items():
            kind = kinds.setdefault(name, [0, 0])
            kind[0] += right
            kind[1] += all_
    return pooled, {name: right / all_ for name, (right, all_) in sorted(kinds.items())}


def certified_right(encoding: Target, kinds: Mapping[str, float]) -> bool:
    """Every kind of the certified answer right on a screen: necessary for its machine to be right."""
    return all(share == 1.0 for name, share in kinds.items() if encoding.certified(name))


def _kinds(kinds: Mapping[str, float]) -> dict[str, float]:
    return {name: round(share, 4) for name, share in kinds.items()}


def _windows(encoding: Target, n: int, cases: int, seed: str) -> tuple[list[Example], list[list[int]]]:
    rng = random.Random(seed)
    drawn = [encoding.draw(n, rng) for _ in range(cases)]
    return [encoding.example(c) for c in drawn], drawn


def windows_drawn(encoding: Target, n: int, cases: int, seed: str) -> list[Example]:
    """The examples of `cases` words of length `n` drawn as a run's screen draws them (its seed's name)."""
    return _windows(encoding, n, cases, seed)[0]


def accuracy(net: Any, encoding: Target, rows: Sequence[Example]) -> dict[str, list[int]]:
    """Each answer kind's [correct, all] (division's quotient by class, 0, undefined, nonzero)."""
    tally: dict[str, list[int]] = {}
    predictions: list[Sequence[int]] = net.classify_rows(net.prepare_dataset(rows))
    for (tokens, labels), predicted in zip(rows, predictions, strict=True):
        for name, label, guess in zip(encoding.kinds(tokens, labels), labels, predicted, strict=True):
            t = tally.setdefault(name, [0, 0])
            t[0] += int(label == guess)
            t[1] += 1
    return tally


def diagnose(net: Any, encoding: Target, settings: Settings) -> dict[str, Any]:
    """Each answer's accuracy and each factor's codes used and shares, at the diagnosis's lengths."""
    tally: dict[str, list[int]] = {}
    samples: list[tuple[tuple[int, ...], Hashable]] = []  # codes, the target's state per window
    for n in settings.lengths:
        rows, drawn = _windows(encoding, n, settings.cases, f"family-diagnose-{n}")
        for name, (c, a) in accuracy(net, encoding, rows).items():
            t = tally.setdefault(name, [0, 0])
            t[0] += c
            t[1] += a
        codes: list[list[tuple[int, ...]]] = net.state_codes(net.backend.matrix([tokens for tokens, _ in rows]))
        for states, cols in zip(codes, drawn, strict=True):
            for w, state in enumerate(encoding.states(cols)):
                samples.append((states[w + 1], state))
    factors = [
        {
            "codes": len({c[g] for c, _ in samples}),
            "shares": {
                q: round(wd.share([(c[g], f(state)) for c, state in samples]), 3)
                for q, f in encoding.quantities.items()
            },
        }
        for g in range(len(samples[0][0]))
    ]
    return {
        "accuracy": {name: round(c / a, 3) for name, (c, a) in sorted(tally.items())},
        "factors": factors,
        "state_share": round(wd.share([(c, state) for c, state in samples]), 3),
    }


def long_accuracy(net: Any, encoding: Target, settings: Settings) -> dict[str, float]:
    """The certified answer's accuracy (division's quotient) over every window at each long length."""
    result: dict[str, float] = {}
    for n in settings.long:
        rows, _ = _windows(encoding, n, settings.cases, f"family-long-{n}")
        tally = accuracy(net, encoding, rows)
        q = [tally[k] for k in tally if encoding.certified(k)]
        result[str(n)] = round(sum(c for c, _ in q) / sum(a for _, a in q), 4)
    return result


def _codes(row: Mapping[str, Any]) -> list[int]:
    return [int(factor["codes"]) for factor in row["factors"]]


def _collapsed(before: Sequence[int] | None, codes: Sequence[int], drop: float) -> bool:
    """The codes in use below `drop` of the diagnosis before's (a collapse, as `trace` marks it)."""
    return before is not None and sum(codes) < drop * sum(before)


def trace(rows: Sequence[Mapping[str, Any]], drop: float = 0.5) -> list[str]:
    """
    A run's log.jsonl, a line per diagnosis: step, epoch (or the step's end), each factor's codes in
    use, the state's share of the pair, each answer's accuracy; marked where the codes in use fall
    below `drop` of the diagnosis before (a collapse: the division workplan's stage 4 trace); and a
    line per fork and per restored best. An epoch's screen score alone (a row without codes) isn't shown.
    """
    lines: list[str] = []
    before: list[int] | None = None
    for row in rows:
        if "fork" in row:
            lines.append(
                f"step {row['step']:>3} epoch {row['epoch']:>3} FORK {row['fork']}: screen {row['screen']:.3f} "
                f"peak {row['peak']:.3f} restored {row['restored']:.3f} rate {row['rate']:.2g}"
            )
            continue
        if "restore" in row:
            lines.append(
                f"step {row['step']:>3} epoch {row['epoch']:>3} RESTORE the step's best: screen {row['restore']:.3f} "
                f"restored {row['restored']:.3f}"
            )
            continue
        if "restore_machine" in row:  # R3's kept machine, restored at a plateau
            lines.append(
                f"step {row['step']:>3} epoch {row['epoch']:>3} RESTORE the kept machine: pairs/code "
                f"{row['restore_machine']:.2f} screen {row['restored']:.3f}"
            )
            continue
        if "factors" not in row:  # an epoch's screen score alone
            continue
        codes = _codes(row)
        answers = " ".join(f"{name} {value:.3f}" for name, value in sorted(row["accuracy"].items()))
        end = f" END {row['end']} min {row.get('minimised')}" if "end" in row else ""
        if "machine" in row:  # pairs per code (the one-code-one-state workplan's R2)
            walked = row["machine"]
            end += f" pairs/code {walked['pairs'] / walked['codes']:.2f} right {walked['right']}"
        mark = " << collapse" if _collapsed(before, codes, drop) else ""
        epoch = row.get("epoch", "-")
        share = row.get("state_share", 0.0)
        lines.append(f"step {row['step']:>3} epoch {epoch:>3} codes {codes} share {share:.3f} {answers}{end}{mark}")
        before = codes
    return lines


@dataclass(frozen=True)
class Outcome:
    """
    A run's end from its log.jsonl: the last step, how it ended, epochs trained in all, minutes; and
    its course: the best its worst answer reached at any diagnosis (`peak`, the accuracy of the
    answer furthest behind) and at which step, its collapses (as `trace` marks them) and its large
    drops (`large_drops`), with how many it never recovered from.
    """

    step: int
    end: str  # certified, or the last step's end (passed, plateau, budget, spent)
    epochs: int
    minutes: float
    peak: float
    peak_step: int
    collapses: int
    drops: int  # large drops (large_drops)
    unrecovered: int  # of them, never back to DROP_FROM


def outcome(rows: Sequence[Mapping[str, Any]], drop: float = 0.5) -> Outcome:
    """
    A run's Outcome: certified at the first step's end that certified (the run stops there). A step
    stopped by a budget (`spent`, `deadline`) logs as its epochs the loop's index at the stop, an
    epoch not trained, so one less is counted for it.
    """
    ends = [row for row in rows if "end" in row]
    last = next((row for row in ends if row.get("certified")), ends[-1] if ends else rows[-1])
    epochs = sum(
        int(row["epochs"]) - (row["end"] in ("spent", "deadline")) for row in ends if row["step"] <= last["step"]
    )
    end = "certified" if last.get("certified") else str(last.get("end", "running"))
    diagnoses = [row for row in rows if "accuracy" in row and row["step"] <= last["step"]]
    best = max(diagnoses, key=_worst, default=last)
    collapses = sum(_collapsed(_codes(before), _codes(row), drop) for before, row in itertools.pairwise(diagnoses))
    falls = large_drops(diagnoses)
    return Outcome(
        int(last["step"]),
        end,
        epochs,
        round(float(last["seconds"]) / 60, 1),
        round(min(best.get("accuracy", {"-": 0.0}).values()), 3),  # the log keeps three decimals
        int(best["step"]),
        collapses,
        len(falls),
        sum(not recovered for _, recovered in falls),
    )


def _anneal_at(rows: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> Mapping[str, Any]:
    """The anneal at a diagnosis: its own, or the last epoch's before it (a step's end logs none)."""
    at = next(i for i, r in enumerate(rows) if r is row)
    none: Mapping[str, Any] = {}
    return next((r["anneal"] for r in reversed(rows[: at + 1]) if "anneal" in r), none)


def crucible_course(rows: Sequence[Mapping[str, Any]], drop: float = 0.5) -> dict[str, Any]:
    """
    A seeded crystallisation's run (D26) from its log.jsonl, for a population's reading (D27): the
    warm-up's end (its step, the crucible's length, each factor's codes in use, effective dimension,
    melting point and transitions there; the warm-up's best screen); the crucible's best state share
    in the cool (the heat's are the warm-up's codes, which the melt erases), its step and coupling; the first diagnosis after it whose codes in use fall below `drop` of the
    best's (the loss: its step, phase, coupling, mean temperature); the first grown step; how it ended
    (outcome's). Observables to set against a population's outcomes, early ones first.
    """
    start = next((row for row in rows if "warm_up_end" in row), None)
    warm = [row for row in rows if row.get("warm") and "screen" in row]
    diagnoses = [row for row in rows if "factors" in row and "accuracy" in row]
    grown = next((int(row["step"]) for row in rows if row.get("crucible") == "grown"), None)
    crucible = [row for row in diagnoses if start is not None and row["step"] > start["step"]]
    crucible = [row for row in crucible if grown is None or row["step"] < grown]
    cooled = [row for row in crucible if _anneal_at(rows, row).get("phase") == "cool"] or crucible
    best = max(cooled, key=lambda row: float(row.get("state_share", 0.0)), default=None)
    lost = None
    if best is not None:
        later = cooled[cooled.index(best) + 1 :]
        lost = next((row for row in later if sum(_codes(row)) < drop * sum(_codes(best))), None)

    def anneal(row: Mapping[str, Any] | None) -> dict[str, Any]:
        a: Mapping[str, Any] = {} if row is None else _anneal_at(rows, row)
        if not a:
            return {}
        couplings: list[float] = a["coupling"]
        temperatures: list[float] = a["temperature"]
        return {
            "phase": a["phase"],
            "coupling": round(sum(couplings) / len(couplings), 3),
            "temperature": float(np.mean(temperatures)),
        }

    end = outcome(rows, drop)
    return {
        "warm_up_end": None if start is None else int(start["step"]),
        "crucible": None if start is None else start["crucible"],
        "warm_screen": max((float(row["screen"]) for row in warm), default=None),
        "in_use": [] if start is None else [int(f["in_use"]) for f in start["warm_up_end"]],
        "effective_dimension": [] if start is None else [int(f["effective_dimension"]) for f in start["warm_up_end"]],
        "melting_point": [] if start is None else [float(f["melting_point"]) for f in start["warm_up_end"]],
        "transitions": [] if start is None else [len(f["transitions"]) for f in start["warm_up_end"]],
        "best_share": None if best is None else float(best.get("state_share", 0.0)),
        "best_step": None if best is None else int(best["step"]),
        "best": anneal(best),
        "lost_step": None if lost is None else int(lost["step"]),
        "lost": anneal(lost),
        "grown": grown,
        "end": end.end,
        "step": end.step,
        "minutes": end.minutes,
    }


def crucible_summary(course: Mapping[str, Any]) -> str:
    """crucible_course's reading on one line."""

    def at(a: Mapping[str, Any]) -> str:
        return f" ({a['phase']}, coupling {a['coupling']:.2f}, T {a['temperature']:.2g})" if a else ""

    best = (
        "-"
        if course["best_share"] is None
        else f"{course['best_share']:.3f} at {course['best_step']}{at(course['best'])}"
    )
    lost = "-" if course["lost_step"] is None else f"{course['lost_step']}{at(course['lost'])}"
    melting = ", ".join(f"{t:.2f}" for t in course["melting_point"])
    return (
        f"warm-up to {course['warm_up_end']} (crucible {course['crucible']}, screen {course['warm_screen']}, "
        f"in use {course['in_use']}, d_eff {course['effective_dimension']}, melting {melting}, "
        f"transitions {course['transitions']}); best share {best}; lost {lost}; grown {course['grown'] or '-'}; "
        f"{course['end']} at {course['step']} ({course['minutes']} min)"
    )


def heat_course(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any] | None]:
    """
    Each factor's cool from a profiled run's log.jsonl (D32's understanding step), its epochs counted
    from the cool's start: its splits (an epoch whose codes' groups rise, D20), the first (nucleation)
    and the largest (the first of the largest rise); its heat capacity's peak (`thermals`, the
    epoch's logged `capacity`), the peak's distance in epochs from each and whether it lies within the
    cascade (from the first split to the last); the capacity's local maxima and its jump across the
    nucleation epoch. None for a factor never soft in the cool, or a log without `capacity`.
    """
    cool = [r["anneal"] for r in rows if r.get("anneal", {}).get("phase") == "cool" and "capacity" in r["anneal"]]
    courses: list[dict[str, Any] | None] = []
    for g in range(len(cool[0]["temperature"]) if cool else 0):
        soft = [a for a in cool if a["temperature"][g] is not None]
        if not soft:
            courses.append(None)
            continue
        T = [float(a["temperature"][g]) for a in soft]
        groups = [int(a["groups"][g]) for a in soft]
        C = [float(a["capacity"][g]) for a in soft]
        splits = [(i, groups[i] - groups[i - 1]) for i in range(1, len(soft)) if groups[i] > groups[i - 1]]
        peak = max(range(len(C)), key=C.__getitem__)
        first = splits[0][0] if splits else None
        largest = max(splits, key=lambda s: s[1])[0] if splits else None
        courses.append(
            {
                "epochs": len(soft),
                "groups": [groups[0], groups[-1]],
                "splits": [[i, rise, round(T[i], 4)] for i, rise in splits],
                "nucleation": first,
                "largest": largest,
                "peak": peak,
                "peak_capacity": C[peak],
                "peak_temperature": round(T[peak], 4),
                "from_nucleation": None if first is None else peak - first,
                "from_largest": None if largest is None else peak - largest,
                "in_cascade": bool(splits) and splits[0][0] <= peak <= splits[-1][0],
                "maxima": sum(1 for i in range(1, len(C) - 1) if C[i - 1] < C[i] > C[i + 1]),
                "jump": None if not first or C[first - 1] <= 0 else round(C[first] / C[first - 1], 2),
            }
        )
    return courses


def heat_summary(courses: Sequence[Mapping[str, Any] | None]) -> str:
    """heat_course's reading on one line."""

    def factor(g: int, c: Mapping[str, Any] | None) -> str:
        if c is None:
            return f"factor {g} -"
        splits = c["splits"]
        span = (
            f"T {splits[0][2]:.3g} to {splits[-1][2]:.3g}, epochs {splits[0][0]}-{splits[-1][0]}" if splits else "none"
        )
        return (
            f"factor {g}: groups {c['groups'][0]} to {c['groups'][1]} in {c['epochs']} epochs, {len(splits)} splits "
            f"({span}; nucleation {c['nucleation']}, largest {c['largest']}, jump x{c['jump']}); capacity peak "
            f"{c['peak_capacity']:.3g} at {c['peak']} (T {c['peak_temperature']:.3g}), {c['from_nucleation']} from "
            f"nucleation, {c['from_largest']} from the largest, {'in' if c['in_cascade'] else 'outside'} the "
            f"cascade; {c['maxima']} local maxima"
        )

    return "; ".join(factor(g, c) for g, c in enumerate(courses))


DROP_FROM, DROP_BY = 0.9, 0.3
RECOVERED = 0.05  # a drop recovered: a later point's worst answer back within this of its level before


def _worst(row: Mapping[str, Any]) -> float:
    return min(row["accuracy"].values())


def large_drops(diagnoses: Sequence[Mapping[str, Any]]) -> list[tuple[int, bool]]:
    """
    A run's large drops (division stage 5's trace): its worst answer from at least DROP_FROM down by
    more than DROP_BY between two diagnoses, each as the step it fell in and whether a later diagnosis
    is back at DROP_FROM.
    """
    falls: list[tuple[int, bool]] = []
    for i, (before, after) in enumerate(itertools.pairwise(diagnoses)):
        if _worst(before) >= DROP_FROM and _worst(after) < _worst(before) - DROP_BY:
            falls.append((int(after["step"]), any(_worst(row) >= DROP_FROM for row in diagnoses[i + 2 :])))
    return falls


@dataclass(frozen=True)
class TriggerScore:
    """A trigger read against one run's log (the p11-trigger workplan, D6)."""

    points: int  # the logged points scored: each epoch's screen with its kinds, or else each diagnosis
    drops: int  # large drops: the worst answer kind from DROP_FROM down by more than DROP_BY between points
    caught: int  # drops at whose point the trigger fires
    unrecovered: int  # drops after which no point is back within RECOVERED of the level before it
    unrecovered_caught: int
    false_alarms: int  # firings near a solution (the worst kind at DROP_FROM before) at no large drop
    early: int  # firings while still learning (the worst kind below DROP_FROM before): a restore, a cooler rate


def score_trigger(rows: Sequence[Mapping[str, Any]], trigger: Trigger) -> TriggerScore:
    """
    A trigger against a run's log, offline: at each logged point past a step's first, its signal
    against the step's peak of it over the points before (D6). A log with each epoch's answer kinds
    (D4) is scored every epoch, an older one at its diagnoses (every few epochs, on the diagnosis's
    cases: a proxy). A fork arm's log is altered by its forks: score runs made without one.
    """
    epochs = [r for r in rows if "kinds" in r and "fork" not in r and r["step"] > 0]
    points = (
        [(r, r["kinds"]) for r in epochs]
        if epochs
        else [(r, r["accuracy"]) for r in rows if "accuracy" in r and isinstance(r.get("epoch"), int) and r["step"] > 0]
    )
    worst = [min(kinds.values()) for _, kinds in points]
    drops = caught = unrecovered = unrecovered_caught = alarms = early = 0
    for i in range(1, len(points)):
        row, kinds = points[i]
        prior = [trigger.value(r.get("screen", 0.0), k) for r, k in points[:i] if r["step"] == row["step"]]
        fires = bool(prior) and trigger.fires(trigger.value(row.get("screen", 0.0), kinds), max(prior))
        if worst[i - 1] >= DROP_FROM and worst[i] < worst[i - 1] - DROP_BY:
            lost = not any(w >= worst[i - 1] - RECOVERED for w in worst[i + 1 :])
            drops, caught = drops + 1, caught + fires
            unrecovered, unrecovered_caught = unrecovered + lost, unrecovered_caught + (fires and lost)
        elif worst[i - 1] >= DROP_FROM:
            alarms += fires
        else:
            early += fires
    return TriggerScore(len(points), drops, caught, unrecovered, unrecovered_caught, alarms, early)


NEAR = 0.95  # a run's worst answer at least this at some diagnosis: it came near a solution
VERDICTS = (wc.SNAP_RATIO, 1.0)  # a certified network's two recertifications: the default snap ratio and strict


def readout(
    root: Path, arms: Sequence[str], certify: Path, pairs: Sequence[tuple[str, str]] | None = None
) -> list[str]:
    """
    A registered comparison's readout (the division workplan's stage 5): each run in root/<arm>/seed<s>
    by its log, a run counted as certified only when its saved network (matched by sha256) has a
    certificate in `certify` in both verdicts (VERDICTS); each arm's count with its Wilson 95%
    interval; for each pair (better, other), Fisher's exact test, one-sided, that the first arm
    certifies more often than the second (by default the first two arms of two), and with more than
    one pair each p Holm-adjusted over the pairs (the approximation claim's D5).
    Each run's course too (Outcome), and how many of an arm's failures came near a solution (NEAR):
    a failure near one lost it, one never near never found it.
    """
    verdicts: dict[str, dict[float, bool]] = {}
    for path in sorted(certify.glob("*.certificate.json")):
        result = json.loads(path.read_text())
        verdicts.setdefault(result["sha256"], {})[result["snap_ratio"]] = bool(result["certified"])
    lines: list[str] = []
    counts: dict[str, tuple[int, int]] = {}
    for arm in arms:
        seeds = sorted((root / arm).glob("seed*/log.jsonl"), key=lambda p: int(p.parent.name.removeprefix("seed")))
        certified, near, collapsed, drops, unrecovered, dropped = 0, 0, 0, 0, 0, 0
        lines.append(f"== {arm}")
        for log in seeds:
            run = outcome([json.loads(line) for line in log.read_text().splitlines()])
            note = ""
            if run.end == "certified":
                network = log.parent / f"certified-step-{run.step}.network.json"
                found = verdicts.get(hashlib.sha256(network.read_bytes()).hexdigest(), {})
                if all(found.get(ratio) for ratio in VERDICTS):
                    certified += 1
                else:
                    note = f"  NOT RECERTIFIED (verdicts found: {found})"
            if run.end != "certified":
                near += run.peak >= NEAR
                collapsed += run.collapses > 0
            drops, unrecovered, dropped = drops + run.drops, unrecovered + run.unrecovered, dropped + (run.drops > 0)
            lines.append(
                f"{log.parent.name}: {run.end} at step {run.step}, {run.epochs} epochs, {run.minutes} min; "
                f"worst answer at best {run.peak:.3f} (step {run.peak_step}), collapses {run.collapses}, "
                f"large drops {run.drops} ({run.unrecovered} unrecovered){note}"
            )
        low, high = inference.wilson(certified, len(seeds))
        lines.append(f"{arm}: {certified} of {len(seeds)} certified (Wilson 95%: {low:.2f}-{high:.2f})")
        failures = len(seeds) - certified
        lines.append(
            f"{arm}'s {failures} failures: {near} came near a solution (worst answer >= {NEAR}), {collapsed} collapsed"
        )
        lines.append(f"{arm}: {drops} large drops in {dropped} runs, {drops - unrecovered} recovered")
        counts[arm] = (certified, len(seeds))
    if pairs is None:
        pairs = [(arms[0], arms[1])] if len(arms) == 2 else []
    ps: list[float] = []
    for better, other in pairs:
        (a, n), (c, m) = counts[better], counts[other]
        ps.append(inference.fisher_exact_greater(a, n - a, c, m - c))
    adjusted = inference.holm(ps)
    for (better, other), p, held in zip(pairs, ps, adjusted, strict=True):
        line = f"Fisher's exact, one-sided ({better} more often than {other}): p {p:.3g}"
        lines.append(line if len(pairs) == 1 else f"{line}, Holm-adjusted over {len(pairs)}: {held:.3g}")
    return lines


STALL_SHARE, STALL_EPOCHS = 0.95, 10  # a guarded run frozen (the guard workplan's D9)


def scaled_factors(q: int, ratio: Fraction, count: int, width: int = 16) -> list[tuple[int, int]]:
    """
    count factors of width, each ceil(ratio sqrt(q)) codes for a target of q states, computed exactly:
    the surplus held at ratio (the division ladder's D2, ceil(ratio p) for division's p * p states;
    the general algebras' D3).
    """
    bound = ratio * ratio * q
    codes = math.isqrt(math.floor(bound))
    while codes * codes < bound:
        codes += 1
    return [(width, codes)] * count


def exclusion_bound(q: int, factors: Sequence[tuple[int, int]]) -> int:
    """`m + q - 1` (experiment-design.md, section 1): the joint codes plus the target's q states, less one."""
    return math.prod(m for _, m in factors) + q - 1


def first_length(machine: wt.Mealy) -> int:
    """
    `s0`, the curriculum's first step (the code-consolidation workplan's D3): the least n at which the
    answers of every training word of lengths 1..n (n tokens, then token 0, as `draw` makes them)
    aren't a function of the token alone, so a network with no state can't pass the first step.
    Exact, from the states reached at each depth: a word's i-th token leaves a state reached in i - 1
    tokens, its final token 0 one reached in n. 1 for a machine whose answers never need the state.
    """
    seen: list[set[int]] = [set() for _ in range(machine.tokens)]  # each token's answers so far
    level = {machine.start}  # the states n - 1 tokens reach
    for n in range(1, machine.states + 2):  # by n = q + 1 every reachable state has been a level
        for s in level:
            for t in range(machine.tokens):
                seen[t].add(int(machine.output[s, t]))
        level = {int(machine.transition[s, t]) for s in level for t in range(machine.tokens)}
        seen[0].update(int(machine.output[s, 0]) for s in level)
        if any(len(answers) > 1 for answers in seen):
            return n
    return 1


def epoch_lengths(step: int, examples: int, long_tail: int | None = None, cap: int | None = None) -> list[int]:
    """
    Each of an epoch's examples' lengths at a step: 1..step in turn; with `long_tail` F (the
    workplan's D4, B), every other example's in turn over step + 1..F step instead, capped at `cap`
    (all of them 1..step when that's empty), so half the examples are the curriculum's as before.
    """
    short = range(1, step + 1)
    if long_tail is None:
        return [short[i % len(short)] for i in range(examples)]
    long = range(step + 1, min(long_tail * step, cap if cap is not None else long_tail * step) + 1)
    if not long:
        return [short[i % len(short)] for i in range(examples)]
    return [short[i // 2 % len(short)] if i % 2 == 0 else long[i // 2 % len(long)] for i in range(examples)]


def whole_lengths(cap: int, points: int = 10) -> list[int]:
    """
    The screen's lengths when every epoch trains on the whole distribution (run's `whole`): 1 to 4 and
    `points` more spaced geometrically to `cap`, its last.
    """
    spread = {1, 2, 3, 4, cap} | {round(cap ** (i / points)) for i in range(points + 1)}
    return sorted(n for n in spread if n <= cap)


def consolidation(net: Any, encoding: Target, d: int) -> tuple[dict[tuple[int, ...], tuple[int, ...]], int, int]:
    """
    A's table from the network alone (the workplan's D1): its reachable joint codes grouped by
    d-equivalence over every slot's outputs, each duplicate to its group's representative (the code
    its shortest word reaches; machine_reading.groups); with the codes reached and the groups.
    """
    walk = mr.walk(QueryMachine(net, encoding, None), encoding.task.machine.tokens)
    found = mr.groups(walk, d)
    return mr.consolidation(walk, found), len(walk.codes), len(found)


def machine_summary(net: Any, encoding: Target) -> dict[str, int]:
    """The network's machine walked beside the target's, in three counts (machine_reading.summary), d the task's."""
    machine = encoding.task.machine
    return mr.summary(mr.product(QueryMachine(net, encoding, 0), machine), mr.word_length(machine))


CHECKPOINT = "checkpoint.json"


@dataclass(frozen=True)
class Thermostat:
    """
    The commitment-control workplan's usage thermostat (stage 2, D1 (a), D2'): each factor's
    commitment off while its effective codes (effective_codes) are below `floor`; once they reach
    `restore` of its codebook, ramped back to its specs' over `ramp` epochs; off again only below
    `floor` (the gap between the two is the hysteresis). A factor's level is 0 (off) to `ramp` (its
    specs' commitment). With `restore_screen` (D2''), a factor off is restored once the step's best
    screen reaches it, in place of half its codebook: the answers right, so a state has formed.
    """

    floor: float = 3.0
    restore: float = 0.5
    ramp: int = 20
    restore_screen: float | None = None

    def levels(
        self, levels: Sequence[int], effective: Sequence[float], counts: Sequence[int], screen: float = 0.0
    ) -> list[int]:
        """Each factor's next level from its level, its effective codes, its codebook's size and the step's best screen."""
        following: list[int] = []
        for level, codes, count in zip(levels, effective, counts, strict=True):
            if codes < self.floor:
                following.append(0)
            elif level > 0:
                following.append(min(level + 1, self.ramp))
            elif self.restore_screen is not None:
                following.append(1 if screen >= self.restore_screen else 0)
            else:
                following.append(1 if codes >= self.restore * count else 0)
        return following

    def commitments(self, levels: Sequence[int], full: Sequence[float]) -> list[float]:
        """Each factor's commitment at its level, a share of its specs'."""
        return [c * level / self.ramp for level, c in zip(levels, full, strict=True)]


def effective_codes(net: Any, rows: Sequence[Sequence[tuple[Any, Any]]]) -> list[float]:
    """
    Each factor's effective codes, 2 ** H of its codes' use over every window's next state of the
    rows given (lists of windows, one length each): the thermostat's signal (the network's own,
    never the target's).
    """
    counts: list[dict[int, int]] = []
    for batch in rows:
        codes: list[list[tuple[int, ...]]] = net.state_codes(net.backend.matrix([tokens for tokens, _ in batch]))
        for states in codes:
            for state in states[1:]:
                if not counts:
                    counts = [{} for _ in state]
                for g, code in enumerate(state):
                    counts[g][code] = counts[g].get(code, 0) + 1
    effective: list[float] = []
    for used in counts:
        total = sum(used.values())
        entropy = -sum(n / total * math.log2(n / total) for n in used.values())
        effective.append(2**entropy)
    return effective


@dataclass(frozen=True)
class Anneal:
    """
    The commitment-control workplan's annealing (stage 3, the soft-to-hard quantiser; D6-D8): each
    snapped factor's codes soft in training at `start` halved every `halving` epochs trained, its
    commitment 0, until `share` of the sample's windows weight one code above `hard`; then soft for
    `tail` epochs more, cooling on (the state formation workplan's D19: the crystal settles), then
    hard for good, its commitment ramped back to its specs' over `ramp` epochs (never, without
    `commit`). A factor's level is 0
    (soft, not yet hard), 1 to `tail` (soft, in its tail), to `tail + ramp` (hard, its specs'
    commitment).
    """

    start: float = 1.0
    halving: int = 10
    hard: float = 0.99
    share: float = 0.99
    ramp: int = 20
    tail: int = 0
    commit: bool = True

    @property
    def top(self) -> int:
        """The last level: hard, at its specs' commitment."""
        return self.tail + self.ramp

    def soft(self, level: int) -> bool:
        """Whether a factor at this level trains soft."""
        return level <= self.tail

    def temperature(self, spent: int) -> float:
        """The soft factors' temperature after `spent` epochs trained in all."""
        return self.start * 0.5 ** (spent // self.halving)

    def levels(self, levels: Sequence[int], hardness: Sequence[float | None]) -> list[int]:
        """Each factor's next level from its level and its hardness (None: continuous, or not read)."""
        following: list[int] = []
        for level, hard in zip(levels, hardness, strict=True):
            if level > 0:
                following.append(min(level + 1, self.top))
            else:
                following.append(1 if hard is not None and hard >= self.share else 0)
        return following

    def commitments(self, levels: Sequence[int], full: Sequence[float]) -> list[float]:
        """Each factor's commitment at its level, a share of its specs' (0 throughout without `commit`)."""
        if not self.commit:
            return [0.0] * len(levels)
        return [c * max(level - self.tail, 0) / self.ramp for level, c in zip(levels, full, strict=True)]


def hardness(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], temperatures: Sequence[float | None], hard: float
) -> list[float | None]:
    """
    Each soft factor's share of every window's cell output, over the rows given (lists of windows, one
    length each), whose largest weight at its temperature (the codes' softmax, as training's) is above
    `hard`; None for a factor at None (annealing's test, D8: the network's own, never the target's).
    """
    starts = [sum(f.state_size for f in net.recurrence.factors[:g]) for g in range(len(net.recurrence.factors))]
    above = [0] * len(temperatures)
    total = 0
    for batch in rows:
        _, _, cells = net.window_states(net.backend.matrix([tokens for tokens, _ in batch]))
        for cell in cells[1:]:
            H = np.array(cell.tolist(), dtype=float)
            total += len(H)
            for g, (factor, t) in enumerate(zip(net.recurrence_layer.factors, temperatures, strict=True)):
                if t is None:
                    continue
                E = np.array(factor.codes.E.tolist(), dtype=float)
                h = H[:, starts[g] : starts[g] + E.shape[1]]
                D = np.sum((h[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2, axis=2)
                A = np.exp(-(D - D.min(axis=1)[:, np.newaxis]) / t)
                above[g] += int(np.sum(A.max(axis=1) / A.sum(axis=1) > hard))
    return [None if t is None else above[g] / max(total, 1) for g, t in enumerate(temperatures)]


def thermals(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], temperatures: Sequence[float | None]
) -> list[tuple[float, float] | None]:
    """
    Each soft factor's energy and heat capacity at its temperature over the rows given (D32's
    understanding step): the mean over windows of the soft assignment's distortion, `<D>` (Rose 1998's
    energy), and the mean of its variance under the assignment over `T^2`, `d<D>/dT` (its specific heat,
    peaking at a phase transition); None for a factor at None. Exact, no sampling; the target unread.
    """
    starts = [sum(f.state_size for f in net.recurrence.factors[:g]) for g in range(len(net.recurrence.factors))]
    energy = [0.0] * len(temperatures)
    spread = [0.0] * len(temperatures)
    total = 0
    for batch in rows:
        _, _, cells = net.window_states(net.backend.matrix([tokens for tokens, _ in batch]))
        for cell in cells[1:]:
            H = np.array(cell.tolist(), dtype=float)
            total += len(H)
            for g, (factor, t) in enumerate(zip(net.recurrence_layer.factors, temperatures, strict=True)):
                if t is None:
                    continue
                E = np.array(factor.codes.E.tolist(), dtype=float)
                h = H[:, starts[g] : starts[g] + E.shape[1]]
                D = np.sum((h[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2, axis=2)
                A = np.exp(-(D - D.min(axis=1)[:, np.newaxis]) / t)
                P = A / A.sum(axis=1)[:, np.newaxis]
                mean = np.sum(P * D, axis=1)
                energy[g] += float(mean.sum())
                spread[g] += float(np.sum(np.sum(P * D**2, axis=1) - mean**2))
    n = max(total, 1)
    return [None if t is None else (energy[g] / n, spread[g] / n / (t * t)) for g, t in enumerate(temperatures)]


def _groups(close: np.ndarray) -> int:
    """How many groups a symmetric closeness matrix joins its items into (transitively)."""
    group = list(range(len(close)))

    def root(i: int) -> int:
        while group[i] != i:
            i = group[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(close, 1)), strict=True):
        group[root(int(i))] = root(int(j))
    return len({root(i) for i in range(len(close))})


def factor_states(net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], live: bool = False) -> list[np.ndarray]:
    """
    Each factor's cell outputs `h` (windows by its state size) over every window of the rows given
    (`live`: continuous factors read unsnapped).
    """
    starts = [sum(f.state_size for f in net.recurrence.factors[:g]) for g in range(len(net.recurrence.factors))]
    blocks: list[list[np.ndarray]] = [[] for _ in starts]
    for batch in rows:
        _, _, cells = net.window_states(net.backend.matrix([tokens for tokens, _ in batch]), live=live)
        for cell in cells[1:]:
            H = np.array(cell.tolist(), dtype=float)
            for g, f in enumerate(net.recurrence.factors):
                blocks[g].append(H[:, starts[g] : starts[g] + f.state_size])
    return [np.concatenate(b) for b in blocks]


def melting_curve(
    h: np.ndarray,
    k: int,
    rng: np.random.Generator,
    per_octave: int = 4,
    octaves: int = 20,
    iterations: int = 30,
    sample: int = 2000,
) -> dict[str, Any]:
    """
    The states' melting curve (the state formation workplan's D21): Rose's deterministic annealing of
    `k` codes on the fixed outputs `h` (at most `sample` of them, drawn), from twice the whole's
    critical temperature `T_c = 2 lambda_max` down `octaves` octaves, `per_octave` temperatures an
    octave, `iterations` soft-assignment updates at each, the codes perturbed by `10^-6 sqrt(T_c)` so
    identical ones can split; at each temperature the codes' groups closer than `T / 100` squared
    (D20's). Returns `T_c`, the curve (temperature, groups) and its transitions (temperature, groups
    before, after), the highest the melting point: where the codes leave the melt. A split is seen
    late, up to about an octave below its temperature: just below a transition the codes part slowly
    (critical slowing down), within the updates a temperature gets.
    """
    if len(h) > sample:
        h = h[rng.choice(len(h), sample, replace=False)]
    critical = 2.0 * float(np.linalg.eigvalsh(np.cov(h, rowvar=False))[-1])
    scale = 1e-6 * math.sqrt(max(critical, 1e-300))
    codes = np.repeat(h.mean(axis=0)[np.newaxis, :], k, axis=0)
    curve: list[list[float]] = []
    transitions: list[list[float]] = []
    for i in range(per_octave * octaves + 1):
        t = 2.0 * critical * 2.0 ** (-i / per_octave)
        if t <= 0.0:
            break
        for _ in range(iterations):
            codes = codes + rng.normal(0.0, scale, size=codes.shape)
            D = np.sum((h[:, np.newaxis, :] - codes[np.newaxis, :, :]) ** 2, axis=2)
            weights = np.exp(-(D - D.min(axis=1)[:, np.newaxis]) / t)
            weights /= weights.sum(axis=1)[:, np.newaxis]
            mass = weights.sum(axis=0)
            codes = np.where(mass[:, np.newaxis] > 0, (weights.T @ h) / np.maximum(mass, 1e-300)[:, np.newaxis], codes)
        groups = _groups(np.sum((codes[:, np.newaxis, :] - codes[np.newaxis, :, :]) ** 2, axis=2) < t / 100)
        if curve and groups != curve[-1][1]:
            transitions.append([t, curve[-1][1], groups])
        curve.append([t, groups])
        if groups == k:
            break
    return {
        "critical_temperature": critical,
        "melting_point": transitions[0][0] if transitions else None,
        "curve": curve,
        "transitions": transitions,
    }


def melting_curves(net: Any, rows: Sequence[Sequence[tuple[Any, Any]]]) -> list[dict[str, Any] | None]:
    """
    Each snapped factor's melting curve over every window's cell output of the rows given (D21; factor
    `g`'s draws from seed `g`, as at the warm-up's end); None unsnapped.
    """
    return [
        None if f.codes is None else melting_curve(h, f.codes.count, np.random.default_rng([g]))
        for g, (f, h) in enumerate(zip(net.recurrence_layer.factors, factor_states(net, rows), strict=True))
    ]


def target_states(machine: wt.Mealy, rows: Sequence[Sequence[tuple[Any, Any]]]) -> np.ndarray:
    """Each window's target state, the machine's after the word's tokens to it, in factor_states' order."""
    blocks: list[np.ndarray] = []
    for batch in rows:
        words = np.array([[int(t) for t in tokens] for tokens, _ in batch], dtype=np.int64)
        state = np.full(len(words), machine.start, dtype=np.int64)
        for j in range(words.shape[1]):
            state = machine.transition[state, words[:, j]]
            blocks.append(state.copy())
    return np.concatenate(blocks)


def state_separation(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], machine: wt.Mealy, sample: int = 3000, seed: int = 0
) -> dict[str, Any]:
    """
    Whether the network's cell outputs `h` determine the target machine's state (the state formation
    workplan's feasibility check: annealing clusters `h`, so it can find the target's states only where
    `h` already separates them). Over every window of the rows given (at most `sample`, drawn), for
    the joint `h` and each factor's: the leave-one-out nearest neighbour's and the nearest target
    centroid's accuracy at naming the window's target state, against the most common state's share
    (chance). Reads the network as set: continuous factors live (set_continuous first) for a warm-up's.
    The target is read, never trained on: a diagnosis.
    """
    states = factor_states(net, rows, live=True)
    targets = target_states(machine, rows)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(targets), min(sample, len(targets)), replace=False)
    targets = np.asarray(targets[chosen], dtype=np.int64)
    blocks = {"joint": np.concatenate(states, axis=1)[chosen]} | {
        f"factor-{g}": h[chosen] for g, h in enumerate(states)
    }
    found, counts = np.unique(targets, return_counts=True)
    values = np.asarray(found, dtype=np.int64)
    separation: dict[str, Any] = {
        "windows": len(targets),
        "states_seen": len(values),
        "chance": float(counts.max() / len(targets)),
    }
    for name, h in blocks.items():
        D = np.sum((h[:, np.newaxis, :] - h[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(D, np.inf)
        nearest: np.ndarray = np.argmin(D, axis=1)
        neighbour = np.count_nonzero(targets[nearest] == targets) / len(targets)
        centroids = np.array([h[targets == v].mean(axis=0) for v in values])
        to_centroid = np.sum((h[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        closest: np.ndarray = np.argmin(to_centroid, axis=1)
        centroid = np.count_nonzero(values[closest] == targets) / len(targets)
        separation[name] = {"neighbour": neighbour, "centroid": centroid}
    return separation


def state_spread(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], merged: float = 1e-6, share: float = 0.99
) -> list[dict[str, Any]]:
    """
    Each factor's spread over every window's cell output `h` of the rows given (the state formation
    workplan's D3: the network's own, never the target's): the top eigenvalue of `h`'s covariance and
    deterministic annealing's critical temperature, twice it (Rose 1998, Theorem 1); `h`'s effective
    dimension, the fewest principal components holding `share` of its variance (Zhao et al. 2026); the
    codes' pairwise squared distances and their groups closer than `merged` (identical codes); and,
    over the windows' nearest codes, the codes in use and the pooled within-cluster covariance's top
    eigenvalue (D7's `tau_0` is twice it). A factor without codes has its codes' entries None.
    """
    states = factor_states(net, rows)
    spread: list[dict[str, Any]] = []
    for g, factor in enumerate(net.recurrence_layer.factors):
        h = states[g]
        eigen = np.sort(np.linalg.eigvalsh(np.cov(h, rowvar=False)))[::-1].clip(min=0.0)
        held = np.cumsum(eigen) / max(float(eigen.sum()), 1e-300)
        row: dict[str, Any] = {
            "windows": len(h),
            "lambda_max": float(eigen[0]),
            "critical_temperature": 2.0 * float(eigen[0]),
            "effective_dimension": int(np.searchsorted(held, share) + 1),
            "codes": None,
            "groups": None,
            "nearest_squared": None,
            "median_squared": None,
            "in_use": None,
            "within_lambda_max": None,
        }
        if factor.codes is not None:
            E = np.array(factor.codes.E.tolist(), dtype=float)
            pairs = np.sum((E[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2, axis=2)
            upper = pairs[np.triu_indices(len(E), 1)]
            nearest = np.argmin(np.sum((h[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2, axis=2), axis=1)
            within = np.zeros((h.shape[1], h.shape[1]))
            for code in np.unique(nearest):
                members = h[nearest == code]
                centred = members - members.mean(axis=0)
                within += centred.T @ centred
            row |= {
                "codes": len(E),
                "groups": _groups(pairs < merged),
                "nearest_squared": float(upper.min()),
                "median_squared": float(np.median(upper)),
                "in_use": len(np.unique(nearest)),
                "within_lambda_max": float(np.linalg.eigvalsh(within / max(len(h) - 1, 1))[-1]),
            }
        spread.append(row)
    return spread


COOLS: dict[str, Callable[[float], float]] = {  # the shaped cools' octaves fallen, a share of all, by share of time
    "concave": lambda u: u**2,
    "convex": lambda u: 1.0 - (1.0 - u) ** 2,
    "s": lambda u: 3 * u**2 - 2 * u**3,
}


@dataclass(frozen=True)
class WarmUp:
    """
    The state formation workplan's option 1 (D4, D6, D10, D14-D18): every snapped factor run continuous
    (its cell's output passed on, its codes unread, its commitment 0) from the run's start until the
    end of the first step whose best screen reaches `screen` and where every snapped factor's distinct
    states seen number `distinct` times its codes, or of its `cap`th step; then each factor's codes set
    by k-means of its cells' outputs on the step's screen (`iterations` Lloyd iterations, k-means++
    seeding from the run's seed), and its temperature profiled (the owner's, from crystallisation):
    heated from `T_hot / 2 ** heat` doubling each epoch, held at `T_hot` for `hold` epochs, then
    cooled exponentially, halved every `halving` epochs, with no floor; `T_hot` is `hot` times the
    whole's critical temperature (Rose 1998: above it every code is at the mean, a melt). Each soft
    epoch moves every soft code by thermal noise of variance `kappa T` (D17). The run's Anneal reads
    hardness from the cool's start and holds the tail and the commitment's ramp. With `gate` (D29) the
    cool is quasi-static: its clock (the profile's epochs) holds through an epoch whose screen before it
    is below `gate` times the hold's best, so the temperature and the coupling wait for the network;
    with `window` (D30) each factor's coupling window spans at least that many octaves below its melting
    point. With `shape` `adaptive` (D32, D33) the cool isn't precomputed: each factor's temperature
    steps once an equilibrium test passes (`within`) or after `settle` epochs at it, by a step sized
    from its heat capacity (`pace`), as Cooling sets it.
    """

    screen: float = 0.95
    cap: int = 8
    iterations: int = 25
    distinct: float = 4.0
    hot: float = 2.0
    heat: int = 6
    hold: int = 10
    halving: int = 15
    kappa: float = 1e-2
    shape: str = "exponential"
    cool: int = 600
    octaves: float = 20.0
    handover: bool = False
    gate: float = 0.0
    window: float = 0.0
    pace: float = 0.03
    settle: int = 10
    within: float = 0.75

    def holds(self, n: int, screen: float, level: float) -> bool:
        """
        Whether the clock holds at `n` (the profile's epochs so far) for the next epoch (D29): within the
        cool, the screen before it (`screen`) below `gate` times the hold's best (`level`).
        """
        return self.gate > 0 and self.phase(n - 1) == "cool" and screen < self.gate * level

    def coupling(self, n: int, temperature: float, melting: float, final: float) -> float:
        """
        A soft factor's coupling after `n` epochs from the warm-up's end at `temperature` (D25, with
        `handover`; else 1): 0 through the heat and the hold (the state the cells' outputs, the codes
        forming on them), then in the cool log-linear in T from 0 at its melting point `melting` to 1
        at its last transition `final` (the melting curve's, D21), clipped to [0, 1].
        """
        if not self.handover:
            return 1.0
        if self.phase(n) != "cool":
            return 0.0
        share = math.log(melting / temperature) / math.log(melting / final)
        return min(max(share, 0.0), 1.0)

    def ends(self, step: int, first: int, best: float, distinct: Sequence[int], counts: Sequence[int]) -> bool:
        """Whether the warm-up ends at this step's end (`distinct`, `counts`: each snapped factor's states and codes)."""
        filled = all(n >= self.distinct * c for n, c in zip(distinct, counts, strict=True))
        return (best >= self.screen and filled) or step - first + 1 >= self.cap

    def reads_hardness(self, n: int) -> bool:
        """
        Whether the hardness test is read after `n` epochs from the warm-up's end: from the cool's start
        (exponential), or once a shaped cool has run its `cool` epochs, so its whole curve trains soft.
        """
        cooled = n - self.heat - self.hold
        return cooled >= 0 and (self.shape in ("exponential", "adaptive") or cooled >= self.cool)

    def phase(self, n: int) -> str:
        """The profile's phase after `n` epochs trained from the warm-up's end."""
        return "heat" if n < self.heat else "hold" if n < self.heat + self.hold else "cool"

    def temperature(self, n: int, hot: float) -> float:
        """
        A factor's temperature after `n` epochs trained from the warm-up's end, at its `T_hot` `hot`. The
        cool's shape on a log-temperature scale (D24): `exponential` a straight line, halved every
        `halving` epochs; `concave` (the octaves fallen go as `u ** 2`, `u` the share of `cool` epochs
        passed: it stays near the melt, then falls fast), `convex` (as `1 - (1 - u) ** 2`: it falls
        fast, then a long slow tail) and `s` (as `3 u ** 2 - 2 u ** 3`: slow near the melt and in the
        tail, fast between), each falling `octaves` in `cool` epochs, then halved every `halving` as
        the exponential; `adaptive` (D33) `hot` here, its cool Cooling's.
        """
        if n < self.heat:
            return hot * 2.0 ** (n - self.heat)
        m = max(n - self.heat - self.hold, 0)
        if self.shape == "adaptive":
            return hot  # the cool's start; Cooling's from there (D33)
        if self.shape == "exponential":
            return hot * 0.5 ** (m / self.halving)
        if m >= self.cool:
            return hot * 2.0**-self.octaves * 0.5 ** ((m - self.cool) / self.halving)
        u = m / self.cool
        return hot * 2.0 ** (-self.octaves * COOLS[self.shape](u))


@dataclass
class Cooling:
    """
    The adaptive cool's state (D32, D33: Huang, Romeo and Sangiovanni-Vincentelli 1986), each soft
    factor's: its temperature, the energies and heat capacities measured at it (`thermals`, an epoch
    each), and the energy predicted for it from the last (None: none, the cool's first). WarmUp's
    `pace` is lambda, `within` the equilibrium test's share of the spread, `settle` the cap.
    """

    temperature: list[float]
    energies: list[list[float]]
    capacities: list[list[float]]
    predicted: list[float | None]

    @staticmethod
    def start(hot: Sequence[float]) -> Cooling:
        """The cool at its start, each factor at its T_hot."""
        return Cooling(list(hot), [[] for _ in hot], [[] for _ in hot], [None] * len(hot))

    def advance(self, warm_up: WarmUp) -> None:
        """
        Each factor's temperature for the next epoch (D33): it steps once the epochs at it pass the
        equilibrium test or number `settle`, to `T exp(-pace / sqrt(C))` (Huang et al.'s `T exp(-lambda
        T / sigma)`, sigma `T sqrt(C)`, C the capacity's mean at T), at most an octave; the energy
        predicted there `E - C dT`. The test: the first epoch's energy within `within` sigma of its
        prediction, or a later epoch's within `within` sigma of the epoch's before (stationary at T).
        """
        for g, (t, energies, capacities) in enumerate(
            zip(self.temperature, self.energies, self.capacities, strict=True)
        ):
            if not energies:
                continue
            capacity = sum(capacities) / len(capacities)
            sigma = t * math.sqrt(max(capacity, 0.0))
            last = energies[-1]
            before = self.predicted[g] if len(energies) == 1 else energies[-2]
            settled = before is not None and abs(last - before) <= warm_up.within * sigma
            if not (settled or len(energies) >= warm_up.settle):
                continue
            ratio = math.exp(-warm_up.pace / math.sqrt(capacity)) if capacity > 0 else 0.0
            cooler = t * max(ratio, 0.5)
            self.temperature[g] = cooler
            self.predicted[g] = last - capacity * (t - cooler)
            self.energies[g], self.capacities[g] = [], []

    def record(self, heat: Sequence[tuple[float, float] | None]) -> None:
        """Each soft factor's energy and capacity this epoch, at its temperature (thermals'; None: hard)."""
        for g, e in enumerate(heat):
            if e is not None:
                self.energies[g].append(e[0])
                self.capacities[g].append(e[1])


def handover_window(made: Sequence[Mapping[str, Any]], window: float = 0.0) -> tuple[list[float], list[float]]:
    """
    Each factor's coupling window from the warm-up end's rows (codes_from_states; D25): its melting
    point and its last transition's temperature, the whole's critical temperature and a thousandth of
    the melting point where its curve has none; at least `window` octaves wide (D30: a curve whose
    codes part in one transition would otherwise hand over within an octave); a factor without codes 1
    and 1 (never coupled).
    """
    melting: list[float] = []
    final: list[float] = []
    for m in made:
        if not m:
            melting.append(1.0)
            final.append(1.0)
            continue
        top = float(m.get("melting_point") or m["critical_temperature"])
        transitions: list[list[float]] = m.get("transitions") or []
        last = float(transitions[-1][0]) if transitions else top / 1000
        melting.append(top)
        final.append(min(last if last < top else top / 2, top * 2.0**-window))
    return melting, final


def distinct_states(net: Any, rows: Sequence[Sequence[tuple[Any, Any]]]) -> list[int]:
    """Each snapped factor's distinct cell outputs (to 6 decimals) over every window of the rows given (WarmUp, D14)."""
    return [
        len(np.unique(np.round(h, 6), axis=0))
        for h, f in zip(factor_states(net, rows), net.recurrence.factors, strict=True)
        if f.codes is not None
    ]


def jitter(net: Any, temperatures: Sequence[float | None], kappa: float, rng: np.random.Generator) -> None:
    """Every soft factor's codes moved by Gaussian noise of variance `kappa T` in all (D17: thermal noise)."""
    for factor, t in zip(net.recurrence_layer.factors, temperatures, strict=True):
        if t is None:
            continue
        codes = np.array(factor.codes.E.tolist(), dtype=float)
        moved = codes + rng.normal(0.0, math.sqrt(kappa * t / codes.shape[1]), size=codes.shape)
        factor.codes.E = net.backend.matrix(moved.tolist())


def code_groups(net: Any, temperatures: Sequence[float | None], kappa: float = 0.0) -> list[int | None]:
    """
    Each soft factor's codes' groups closer than `T max(1 / 100, 10 kappa)` squared (D20: the melt and
    its splits, the distance above thermal noise's `kappa T`); None if hard.
    """
    groups: list[int | None] = []
    for factor, t in zip(net.recurrence_layer.factors, temperatures, strict=True):
        if t is None:
            groups.append(None)
            continue
        E = np.array(factor.codes.E.tolist(), dtype=float)
        close = t * max(1 / 100, 10 * kappa)
        groups.append(_groups(np.sum((E[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2, axis=2) < close))
    return groups


def kmeans(points: np.ndarray, k: int, rng: np.random.Generator, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """
    k centres of the points (n, d), k-means++ seeding then Lloyd's iterations, and each point's nearest
    centre. A centre with no points (fewer distinct points than k) is a drawn point moved by a small
    random step, so no two codes start equal (deterministic annealing's perturbation).
    """
    n = len(points)
    centres = [points[int(rng.integers(n))]]
    for _ in range(1, k):
        d = np.min(np.sum((points[:, np.newaxis, :] - np.array(centres)[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        total = float(d.sum())
        centres.append(points[int(rng.choice(n, p=d / total))] if total > 0 else points[int(rng.integers(n))])
    C = np.array(centres, dtype=float)
    scale = max(float(np.sqrt(np.var(points, axis=0).sum())), 1e-6)
    for _ in range(iterations + 1):
        nearest = np.argmin(np.sum((points[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        for j in range(k):
            members = points[nearest == j]
            C[j] = (
                members.mean(axis=0)
                if len(members)
                else points[int(rng.integers(n))] + rng.normal(0.0, 1e-3 * scale, size=points.shape[1])
            )
    nearest = np.argmin(np.sum((points[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2), axis=1)
    return C, nearest


def codes_from_states(
    net: Any, rows: Sequence[Sequence[tuple[Any, Any]]], rng: np.random.Generator, iterations: int, hot: float
) -> list[dict[str, Any]]:
    """
    Each snapped factor's codes set by k-means of its cells' outputs over the rows given (WarmUp, D6),
    and its `T_hot`, `hot` times the whole's critical temperature (D16); a row each, with the clusters'
    own critical temperature (twice the pooled within-cluster top eigenvalue), the whole's, and the
    states' melting curve's melting point and transitions (D21) beside it.
    """
    spread = state_spread(net, rows)
    states = factor_states(net, rows)
    made: list[dict[str, Any]] = []
    for g, factor in enumerate(net.recurrence_layer.factors):
        if factor.codes is None:
            made.append({})
            continue
        h = states[g]
        melting = melting_curve(h, factor.codes.count, np.random.default_rng([g]))  # k-means' draws untouched
        E, nearest = kmeans(h, factor.codes.count, rng, iterations)
        within = np.zeros((h.shape[1], h.shape[1]))
        for code in np.unique(nearest):
            centred = h[nearest == code] - h[nearest == code].mean(axis=0)
            within += centred.T @ centred
        lam = float(np.linalg.eigvalsh(within / max(len(h) - 1, 1))[-1])
        factor.codes.E = net.backend.matrix(E.tolist())
        made.append(
            {
                "hot": hot * spread[g]["critical_temperature"],
                "within_temperature": 2.0 * lam,
                "critical_temperature": spread[g]["critical_temperature"],
                "in_use": len(np.unique(nearest)),
                "effective_dimension": spread[g]["effective_dimension"],
                "melting_point": melting["melting_point"],
                "transitions": melting["transitions"],
            }
        )
    return made


@dataclass
class Place:
    """
    Where a run is, as its checkpoint keeps it (run's resume): the step and the last epoch trained
    in it (0: the step just begun), the epochs trained in all, the step's best screen and epochs
    since, its rate, forks and the fork's peak, the last screen's answer kinds, the step's undone
    updates and the guard's frozen epochs; the seconds trained, and the log's length in bytes; and
    A's table, (duplicate, representative) pairs (training state, not saved with the network); and
    each factor's levels under the thermostat or annealing.
    """

    step: int
    epoch: int
    spent: int
    best: float
    since: int
    rate: float
    forks: int
    peak: float
    last_kinds: dict[str, float]
    step_rollbacks: int
    frozen: int
    seconds: float
    log_bytes: int
    consolidation: list[tuple[list[int], list[int]]] = field(default_factory=list[tuple[list[int], list[int]]])
    thermostat: list[int] = field(default_factory=list[int])  # each factor's level (Thermostat), with one
    machine_best: list[float] = field(default_factory=list[float])  # R3: the step's kept machine's key, with it
    words: list[list[int]] = field(default_factory=list[list[int]])  # R4: the last diagnosis's counterexamples
    anneal: list[int] = field(default_factory=list[int])  # each factor's level (Anneal), with it
    warm: bool = False  # WarmUp: still continuous
    hot: list[float] = field(default_factory=list[float])  # WarmUp: each factor's T_hot, once set
    anneal_from: int = 0  # WarmUp: the epochs trained when the warm-up ended
    melting: list[float] = field(default_factory=list[float])  # WarmUp: each factor's melting point (D25)
    final: list[float] = field(default_factory=list[float])  # WarmUp: each factor's last transition (D25)
    crucible: int = 0  # the crucible's longest length, once its warm-up ended (D26)
    grown: int = 0  # the crucible's first block of growth, once begun (D26)
    held: int = 0  # WarmUp: the epochs its clock held (D29)
    level: float = -1.0  # WarmUp: the hold's best screen (D29)
    gate_screen: float = -1.0  # WarmUp: the last screen, which the gate reads (D29)
    cooling: dict[str, Any] = field(default_factory=dict[str, Any])  # WarmUp: the adaptive cool's state (D33)
    couplings: list[float] = field(default_factory=list[float])  # WarmUp: the network's last set (D25)


def _run_identity(
    encoding: Target,
    factors: Sequence[tuple[int, int]],
    seed: int,
    budget_seconds: float,
    settings: Settings,
    cap: int,
    reads: str,
    budget_epochs: int | None,
    fork: Fork | None,
    guard: bool,
    first: int,
    long_tail: int | None,
    consolidate: bool,
    restore_best: bool,
    commitment_from: int | None,
    width: int = wr.D,
    branches: bool = False,
    answer_branch: bool = False,
    thermostat: Thermostat | None = None,
    keep_machine: bool = False,
    counterexamples: float | None = None,
    anneal: Anneal | None = None,
    warm_up: WarmUp | None = None,
    whole: bool = False,
    crucible: bool = False,
) -> dict[str, Any]:
    # what a resumed run must share with the run it resumes; the consolidation workplan's settings
    # only when not their defaults, so a checkpoint from before them still resumes
    return {
        "target_states": encoding.target_states,
        "tokens": encoding.tokens,
        "factors": [list(f) for f in factors],
        "seed": seed,
        "budget_seconds": budget_seconds,
        "settings": repr(settings),
        "cap": cap,
        "reads": reads,
        "budget_epochs": budget_epochs,
        "fork": None if fork is None else repr(fork),
        "guard": guard,
        **({"first": first} if first != 1 else {}),
        **({"long_tail": long_tail} if long_tail is not None else {}),
        **({"consolidate": True} if consolidate else {}),
        **({"restore_best": True} if restore_best else {}),
        **({"commitment_from": commitment_from} if commitment_from is not None else {}),
        **({"width": width} if width != wr.D else {}),
        **({"branches": True} if branches else {}),
        **({"answer_branch": True} if answer_branch else {}),
        **({"thermostat": repr(thermostat)} if thermostat is not None else {}),
        **({"keep_machine": True} if keep_machine else {}),
        **({"counterexamples": counterexamples} if counterexamples is not None else {}),
        **({"anneal": repr(anneal)} if anneal is not None else {}),
        **({"warm_up": repr(warm_up)} if warm_up is not None else {}),
        **({"whole": True} if whole else {}),
        **({"crucible": True} if crucible else {}),
    }


COUNTEREXAMPLE_WORDS = 256  # R4's words at most (D6)
BEGUN, ENDED = "begun", "ended"  # a checkpoint's place before the run's first row, and after its last


def _write_checkpoint(out: Path, identity: dict[str, Any], place: Place | str, net: Any) -> None:
    # the run's place (or BEGUN, ENDED) and its network in one file, replaced whole, so a stop at any
    # moment leaves the last checkpoint or the new one
    state = {
        "run": identity,
        "place": place if isinstance(place, str) else vars(place),
        "network": network_to_json(net),
    }
    partial = out / (CHECKPOINT + ".partial")
    partial.write_text(json.dumps(state))
    partial.replace(out / CHECKPOINT)


def run(
    encoding: Target,
    factors: Sequence[tuple[int, int]],
    seed: int,
    budget_seconds: float,
    out: Path,
    settings: Settings = PROBE,
    cap: int | None = None,
    reads: str = BEFORE_STATE,
    budget_epochs: int | None = None,
    fork: Fork | None = None,
    guard: bool = False,
    resume: bool = False,
    long_tail: int | None = None,
    consolidate: bool = False,
    restore_best: bool = False,
    commitment_from: int | None = None,
    width: int = wr.D,
    branches: bool = False,
    answer_branch: bool = False,
    thermostat: Thermostat | None = None,
    keep_machine: bool = False,
    counterexamples: float | None = None,
    anneal: Anneal | None = None,
    warm_up: WarmUp | None = None,
    whole: bool = False,
    crucible: bool = False,
) -> None:
    """
    One probe run: the curriculum from its first step `s0` (first_length: 1 but where a stateless
    network could pass it; a step named by its longest length) until the budget (wall-clock seconds
    and, if given, epochs trained in all, whichever is spent first: the epochs don't depend on the
    machine or its load) or the cap (by default the cell's exclusion bound), a line of JSON per diagnosis to
    out/log.jsonl, the network saved at each step's end to out/network.json. With `fork`, a collapse
    within a step goes back to the step's best network and forks (Fork; a log line each). With
    `guard`, the certificate guard on the certified answer against its task's machine
    (certificate_guard), armed for an epoch whose screen before it got every one right (the guard
    workplan's D5);
    each epoch's row logs whether it was armed, its updates and those undone, each step's end the
    undone's sum. A run whose guard undoes STALL_SHARE of the updates of STALL_EPOCHS armed epochs in
    a row ends "stalled" (D9), its step's network certified as any step's.

    The code-consolidation workplan's arms: `long_tail` F (B, D4) trains every other example at
    lengths step + 1..F step, capped at `cap` (epoch_lengths); `consolidate` (A, D1) reads the
    network's groups at each diagnosis, at D2's `d`, and holds each duplicate code's states to its
    representative's codes until the next (network.consolidate), a fork's restored network
    regrouped; each such row logs `consolidation`, the codes reached, the groups and the duplicates
    pulled. `restore_best` (stage 2c's P2) saves the network at each new best screen of a step
    (out/best.network.json) and, when the step ends on a plateau, restores it before the step's end
    (its save and certificate) and the next step, regrouped with A, as a fork restores its snapshot;
    a row logs the step's best screen (`restore`) and the restored network's (`restored`).
    `commitment_from` J (the commitment-control workplan's diagnostic, D1 (b)) trains every step
    before J with each factor's commitment 0 and from J with its specs' (set before each epoch, so a
    fork's or a restore's loaded network has it too). `anneal` (Anneal) sets each snapped factor's
    temperature and commitment before each epoch from its level, read on the step's screen at its
    shortest, middle and longest lengths, a row's `anneal` logging them. `warm_up` (WarmUp, with `anneal`; the
    state formation workplan's option 1) runs every snapped factor continuous from the start, each
    row logging `warm`, each warm step's end its distinct states (`distinct`), until a step's end
    where it ends: its codes set from the states seen, a row logging `warm_up_end`, then each factor's
    temperature profiled from its T_hot (heat, hold, cool), its soft codes moved by thermal noise,
    each row's `anneal` logging the phase and the codes' groups; no step's best is kept during the
    heat and the hold, and nothing goes back (no fork, restore_best or keep_machine: D18). The
    machine's readings, its certificate included, read the snapped network throughout.
    `whole` (the state formation workplan's D22) drops the curriculum: every epoch's examples drawn
    over every length 1..cap in turn, every step's screen over whole_lengths(cap); a step is then a
    block of epochs ending, as any, in its certificate, and blocks go on past the cap until the budget
    (wall-clock or epochs) or a certificate ends the run.
    `crucible` (with `warm_up`; the state formation workplan's D26, a seed grown in its own crucible)
    holds the material at the warm-up's end: every block after it trains and screens the warm-up's
    last step's lengths, so the whole profile crystallises the state the curriculum formed on one
    material; once every snapped factor is hard, the seed grows: every block from the next on is the
    whole distribution's (as `whole`), its codes hard, the forward path coupled to them.

    Each step's start and each epoch's end that the step goes on from leave out/checkpoint.json:
    the run's place (Place) and its network, with its optimizer's and generator's state. With
    `resume`, a run stopped part way goes on from its checkpoint (the code-consolidation workplan's
    stage 3): its log cut back to the checkpoint's rows, an epoch begun after it trained again, so
    every network and row but the seconds is the uninterrupted run's; the wall-clock budget counts
    the seconds trained. A run that ended isn't trained again; one without a checkpoint starts.
    """
    if cap is None:
        cap = exclusion_bound(encoding.target_states, factors)
    out.mkdir(parents=True, exist_ok=True)
    first = first_length(encoding.task.machine)
    identity = _run_identity(
        encoding,
        factors,
        seed,
        budget_seconds,
        settings,
        cap,
        reads,
        budget_epochs,
        fork,
        guard,
        first,
        long_tail,
        consolidate,
        restore_best,
        commitment_from,
        width,
        branches,
        answer_branch,
        thermostat,
        keep_machine,
        counterexamples,
        anneal,
        warm_up,
        whole,
        crucible,
    )
    assert warm_up is None or anneal is not None, "the warm-up's end is an annealed snap: with an Anneal"
    assert not (whole and long_tail is not None), "the whole distribution has no long tail beyond it"
    assert not crucible or (warm_up is not None and not whole and long_tail is None), (
        "a crucible holds a warm-up's material, then grows on the whole distribution: with a warm-up, not --whole"
    )
    assert warm_up is None or warm_up.shape in ("exponential", "adaptive") or warm_up.shape in COOLS, (
        f"a cool's shape: exponential, adaptive or one of {sorted(COOLS)}"
    )
    assert warm_up is None or (fork is None and not restore_best and not keep_machine), (
        "a crystallising run never goes back (D18): the warm-up not with the fork, restore_best or keep_machine"
    )
    assert not (keep_machine and restore_best), "the step's kept machine (R3) or its best screen, not both"
    assert counterexamples is None or 0 < counterexamples < 1, (
        f"a share of each epoch's examples; got {counterexamples}"
    )
    assert thermostat is None or (commitment_from is None and not consolidate), (
        "the thermostat sets the commitment's weight itself: not with --commitment-from, nor with A, whose target "
        "it moves (the commitment-control workplan)"
    )
    assert anneal is None or (thermostat is None and commitment_from is None and not consolidate), (
        "annealing sets the commitment's weight itself, and its soft codes take no consolidation's target"
    )
    d = mr.word_length(encoding.task.machine) if consolidate else 0
    table: dict[tuple[int, ...], tuple[int, ...]] = {}  # A's, from the last diagnosis
    place: Place | None = None
    saved = json.loads((out / CHECKPOINT).read_text()) if resume and (out / CHECKPOINT).exists() else None
    if saved is not None:
        assert saved["run"] == identity, f"{out} holds another run: {saved['run']} against {identity}"
        if saved["place"] == ENDED:
            return
    if saved is not None and saved["place"] != BEGUN:
        place = Place(**saved["place"])
        net = network_from_state(saved["network"])
        table = {tuple(duplicate): tuple(representative) for duplicate, representative in place.consolidation}
        net.consolidate(table)
        with (out / "log.jsonl").open("r+") as kept:
            kept.truncate(place.log_bytes)
    else:
        assert saved is not None or not (resume and (out / "log.jsonl").exists()), (
            f"{out} has a log but no checkpoint to resume from"
        )
        net = network(encoding, factors, seed, reads, width, branches, answer_branch)
        if saved is not None:
            (out / "log.jsonl").write_text("")  # stopped before its first checkpoint: begun again from the seed
        _write_checkpoint(out, identity, BEGUN, net)
    start = time.perf_counter() - (place.seconds if place is not None else 0.0)
    log = (out / "log.jsonl").open("a")

    def record(row: dict[str, Any]) -> None:
        row["seconds"] = round(time.perf_counter() - start, 1)
        log.write(json.dumps(row) + "\n")
        log.flush()

    if place is None:
        record({"step": 0, "epoch": 0, **diagnose(net, encoding, settings)})
    task = encoding.task
    guarding = cg.Guard(lambda n: QueryMachine(n, encoding, 0), task) if guard else None
    frozen = (
        place.frozen if place is not None else 0
    )  # consecutive armed epochs with STALL_SHARE of their updates undone
    spent = place.spent if place is not None else 0  # epochs trained in all
    snapshot = out / "snapshot.network.json"
    step_best = out / "best.network.json"  # restore_best's
    commitments = [0.0 if f.codes is None else f.codes.commitment for f in net.recurrence.factors]  # the specs'
    counts = [0 if f.codes is None else f.codes.count for f in net.recurrence.factors]
    levels = list(place.thermostat) if place is not None and place.thermostat else [0] * len(counts)  # off at the start
    snapped = [f.codes is not None for f in net.recurrence.factors]
    counts_snapped = [c for c, s in zip(counts, snapped, strict=True) if s]
    annealed = (  # Anneal's levels: soft at the start (a continuous factor's at the ramp's end, never soft)
        list(place.anneal)
        if place is not None and place.anneal
        else [0 if s else (anneal.top if anneal is not None else 0) for s in snapped]
    )
    warm = place.warm if place is not None else warm_up is not None  # WarmUp: still continuous
    hot = list(place.hot) if place is not None else []  # WarmUp: each factor's T_hot, once set
    anneal_from = place.anneal_from if place is not None else 0  # WarmUp: the epochs trained at its end
    melting_points = list(place.melting) if place is not None else []  # WarmUp: each factor's melting point (D25)
    last_splits = list(place.final) if place is not None else []  # WarmUp: each factor's last transition (D25)
    fixed = place.crucible if place is not None else 0  # D26: the crucible's longest length, once set
    grown_from = place.grown if place is not None else 0  # D26: the first block of growth, once begun
    held = place.held if place is not None else 0  # D29: the epochs the profile's clock held
    level = place.level if place is not None else -1.0  # D29: the hold's best screen
    gate_screen = place.gate_screen if place is not None else -1.0  # D29: the last screen
    cooling = Cooling(**place.cooling) if place is not None and place.cooling else None  # D33: the adaptive cool
    coupled = list(place.couplings) if place is not None else []  # D25: the couplings last set, training state
    if coupled:
        net.set_couplings(coupled)  # the loaded network's, as the stopped run's was (its screens read them)

    def warming() -> None:
        # the network about to train or screen (a fork's or a restore's loaded one included) continuous
        # while warming, snapped after (WarmUp)
        if warm_up is not None:
            net.set_temperatures([None] * len(snapped))
            net.set_continuous(list(snapped) if warm else [False] * len(snapped))

    walked_d = mr.word_length(encoding.task.machine)  # R3 and R4's walks' d, the task's
    words: list[list[int]] = [list(w) for w in place.words] if place is not None else []
    machine_kept = out / "machine-best.network.json"
    machine_best: list[float] = []

    def checkpoint(step: int, epoch: int) -> None:
        # the run's place after epoch of step (0: its start), every count read as it stands
        log.flush()
        place = Place(
            step,
            epoch,
            spent,
            best,
            since,
            rate,
            forks,
            peak,
            dict(last_kinds),
            step_rollbacks,
            frozen,
            time.perf_counter() - start,
            log.tell(),
            [(list(duplicate), list(representative)) for duplicate, representative in table.items()],
            list(levels) if thermostat is not None else [],
            list(machine_best),
            [list(w) for w in words],
            list(annealed) if anneal is not None else [],
            warm,
            list(hot),
            anneal_from,
            list(melting_points),
            list(last_splits),
            fixed,
            grown_from,
            held,
            level,
            gate_screen,
            asdict(cooling) if cooling is not None else {},
            list(coupled),
        )
        _write_checkpoint(out, identity, place, net)

    def regroup() -> dict[str, int]:
        # A's table read from the network as it stands and held until the next; the row's record of it
        nonlocal table
        table, codes, groups = consolidation(net, encoding, d)
        net.consolidate(table)
        return {"codes": codes, "groups": groups, "duplicates": len(table)}

    # the curriculum's steps end at the cap; the whole distribution's blocks go on until the budget (D22:
    # a step is then a block of epochs, not a length)
    begin = place.step if place is not None else first
    for step in itertools.count(begin) if whole or crucible else range(begin, cap + 1):
        # the curriculum's lengths, or the whole distribution's every step (a step then a block of epochs);
        # a crucible's (D26) its warm-up's last step's every block, then, every factor hard, the whole's
        # growth begins at the first block to start with every factor hard, and is kept (a resumed block
        # within it reads it from its checkpoint, not from the levels the block's epochs moved)
        if (
            not grown_from
            and not (place is not None and place.step == step)
            and fixed > 0
            and anneal is not None
            and all(not anneal.soft(level) for level, s in zip(annealed, snapped, strict=True) if s)
        ):
            grown_from = step
        grown = grown_from > 0
        flat = whole or grown
        longest = fixed or step
        lengths = whole_lengths(cap) if flat else list(range(1, longest + 1))
        named = "whole" if flat else longest
        screens = {n: _windows(encoding, n, settings.screen, f"family-screen-{named}-{n}")[0] for n in lengths}
        read_at = (  # the step's screen at its shortest, middle and longest
            sorted({lengths[0], lengths[len(lengths) // 2], lengths[-1]})
            if flat
            else sorted({1, (longest + 1) // 2, longest})
        )
        stop, epoch = "budget", 0
        score = -1.0  # the last epoch's screen (WarmUp's end reads it with the step's best)
        if place is not None and place.step == step:  # resumed within this step
            warming()  # the loaded network continuous while warming, as the stopped run's was
            best, since, epoch = place.best, place.since, place.epoch
            rate, forks, peak = place.rate, place.forks, place.peak
            last_kinds, step_rollbacks = place.last_kinds, place.step_rollbacks
            machine_best = list(place.machine_best)
        else:
            best, since = -1.0, 0
            machine_best = []
            rate, forks = settings.rate, 0
            warming()
            start_score, start_kinds = screen(
                net, encoding, screens
            )  # the step before's network on this step's lengths
            record(
                {
                    "step": step,
                    "epoch": 0,
                    "screen": round(start_score, 4),
                    "kinds": _kinds(start_kinds),
                    **({"crucible": "grown" if grown else fixed} if crucible and fixed else {}),
                }
            )
            last_kinds, step_rollbacks = start_kinds, 0
            gate_screen = start_score  # D29: the gate reads the screen before the epoch
            peak = 0.0
            if fork is not None:
                peak = fork.trigger.value(start_score, start_kinds)
                net.save(str(snapshot))

        if epoch == 0:
            checkpoint(step, 0)
        resumed = epoch
        for epoch in range(resumed + 1, settings.max_epochs + 1):
            if time.perf_counter() - start > budget_seconds:
                stop = "deadline"
                break
            if budget_epochs is not None and spent >= budget_epochs:
                stop = "spent"
                break
            rng = random.Random(f"family-train-{seed}-{step}-{epoch}" + (f"-fork{forks}" if forks else ""))
            examples = [
                encoding.example(encoding.draw(n, rng))
                for n in epoch_lengths(cap if flat else longest, settings.epoch_examples, long_tail, cap)
            ]
            if counterexamples is not None and words:
                # R4 (D6): a share of the epoch from the last diagnosis's counterexamples, each word then d
                # uniform tokens and the final zero column (encoding.draw), so its answers tell its states apart
                taken = round(counterexamples * len(examples))
                examples[:taken] = [
                    encoding.example(words[i % len(words)] + encoding.draw(walked_d, rng)) for i in range(taken)
                ]
            if commitment_from is not None:
                net.set_commitments([0.0 if step < commitment_from else c for c in commitments])
            thermo: dict[str, Any] = {}
            melting = False  # WarmUp: in the profile's heat or hold (D18)
            phase: str | None = None  # WarmUp: the profile's phase this epoch trains in
            if thermostat is not None:
                # the network about to train (a fork's restored one included), read on the step's screen at its
                # shortest, middle and longest lengths
                effective = effective_codes(net, [screens[n] for n in read_at])
                levels = thermostat.levels(levels, effective, counts, best)  # best: the step's best screen so far
                beta = thermostat.commitments(levels, commitments)
                net.set_commitments(beta)
                thermo = {
                    "thermostat": {"effective": [round(e, 2) for e in effective], "beta": [round(b, 4) for b in beta]}
                }
            if warm:
                # WarmUp: every snapped factor continuous, its commitment 0 (D4, D9)
                warming()
                net.set_commitments([0.0] * len(snapped))
                thermo = {"warm": True}
            elif anneal is not None:
                # the network about to train (a fork's restored one included), its soft factors' hardness read
                # at the temperature they trained at, on the step's screen at its shortest, middle and longest;
                # after a warm-up, each factor's temperature its profile's from its T_hot (D15), hardness read
                # from the cool's start or a shaped cool's end (D24), the soft codes moved by thermal noise (D17)
                warming()
                holding = (
                    warm_up is not None and bool(hot) and warm_up.holds(spent - anneal_from - held, gate_screen, level)
                )
                if holding:
                    held += 1  # D29: this epoch trains at the last one's clock
                clock = spent - anneal_from - held  # the profile's epochs (D29: less those held)
                phase = warm_up.phase(clock) if warm_up is not None and hot else None
                melting = phase in ("heat", "hold")
                reading = phase is None or (warm_up is not None and warm_up.reads_hardness(clock))
                each = (
                    [warm_up.temperature(clock, t) for t in hot]
                    if warm_up is not None and hot
                    else [anneal.temperature(spent)] * len(snapped)
                )
                if cooling is not None and warm_up is not None and phase == "cool":
                    if not holding:  # D33: the adaptive cool steps from its measurements; D29's gate holds it too
                        cooling.advance(warm_up)
                    each = list(cooling.temperature)
                soft = [
                    t if s and level == 0 and reading else None
                    for t, s, level in zip(each, snapped, annealed, strict=True)
                ]
                hard = hardness(net, [screens[n] for n in read_at], soft, anneal.hard)
                annealed = anneal.levels(annealed, hard)
                temperatures = [
                    t if s and anneal.soft(level) else None for t, s, level in zip(each, snapped, annealed, strict=True)
                ]
                beta = anneal.commitments(annealed, commitments)
                profiled: dict[str, Any] = {}
                couplings = [1.0] * len(snapped)
                if phase is not None and warm_up is not None:
                    jitter(net, temperatures, warm_up.kappa, np.random.default_rng([seed, step, epoch, forks]))
                    profiled = {"phase": phase, "groups": code_groups(net, temperatures, warm_up.kappa)}
                    # D32's logging, and the adaptive cool's reading (D33)
                    heat = thermals(net, [screens[n] for n in read_at], temperatures)
                    profiled["energy"] = [None if e is None else round(e[0], 6) for e in heat]
                    profiled["capacity"] = [None if e is None else round(e[1], 6) for e in heat]
                    if cooling is not None and phase == "cool":  # D33: the measurements the next step reads
                        cooling.record(heat)
                        profiled["at"] = [len(e) for e in cooling.energies]
                    if warm_up.handover:  # D25: each soft factor's coupling from its temperature
                        couplings = [
                            1.0 if t is None else warm_up.coupling(clock, t, m, f)
                            for t, m, f in zip(temperatures, melting_points, last_splits, strict=True)
                        ]
                        profiled["coupling"] = [round(c, 4) for c in couplings]
                    if warm_up.gate:
                        profiled["held"] = held
                net.set_temperatures(temperatures)
                net.set_couplings(couplings)
                coupled = couplings
                net.set_commitments(beta)
                thermo = {
                    "anneal": {
                        "temperature": temperatures,
                        "hard": [None if h is None else round(h, 4) for h in hard],
                        "beta": [round(b, 4) for b in beta],
                        **profiled,
                    }
                }
            guarded: dict[str, Any] = {}
            if guarding is not None:
                if certified_right(encoding, last_kinds):
                    guarding.arm(net)
                else:
                    guarding.disarm()
                updates, rollbacks = guarding.updates, guarding.rollbacks
                wr.train_epoch(net, examples, settings.batch_size, rate, rng, guarding)
                updates, rollbacks = guarding.updates - updates, guarding.rollbacks - rollbacks
                guarded = {"armed": guarding.armed, "updates": updates, "rollbacks": rollbacks}
                step_rollbacks += rollbacks
                frozen = frozen + 1 if updates and rollbacks >= STALL_SHARE * updates else 0
            else:
                wr.train_epoch(net, examples, settings.batch_size, rate, rng)
            spent += 1
            score, kinds = screen(net, encoding, screens)
            gate_screen = score
            if phase == "hold":
                level = max(level, score)  # D29: the liquid's level, the gate's reference
            last_kinds = kinds
            if fork is not None:
                value = fork.trigger.value(score, kinds)
                if value > peak:
                    peak = value
                    net.save(str(snapshot))
                elif fork.trigger.fires(value, peak) and forks < fork.limit:
                    forks += 1
                    rate *= fork.cool
                    net = load_network(str(snapshot))
                    restored, last_kinds = screen(net, encoding, screens)
                    grouped = {"consolidation": regroup()} if consolidate else {}
                    record(
                        {
                            "step": step,
                            "epoch": epoch,
                            "fork": forks,
                            "screen": round(score, 4),
                            "kinds": _kinds(kinds),
                            "peak": round(peak, 4),
                            "restored": round(restored, 4),
                            "rate": rate,
                            **guarded,
                            **thermo,
                            **grouped,
                        }
                    )
                    checkpoint(step, epoch)
                    continue
            if epoch % settings.diagnose_every == 0:
                walked: dict[str, Any] = {}
                if keep_machine or counterexamples is not None:
                    walk = mr.product(QueryMachine(net, encoding, 0), encoding.task.machine)
                    summary = mr.summary(walk, walked_d)
                    walked["machine"] = summary
                    if keep_machine:  # R3 (D5): the fewest pairs per code, ties to more right pairs
                        key = [summary["pairs"] / summary["codes"], -float(summary["right"])]
                        if not machine_best or key < machine_best:
                            machine_best = key
                            net.save(str(machine_kept))
                            walked["kept"] = True
                    if counterexamples is not None:
                        words = mr.counterexamples(walk, walked_d, COUNTEREXAMPLE_WORDS)
                        walked["counterexamples"] = len(words)
                record(
                    {
                        "step": step,
                        "epoch": epoch,
                        "screen": round(score, 4),
                        "kinds": _kinds(kinds),
                        **guarded,
                        **thermo,
                        **diagnose(net, encoding, settings),
                        **({"consolidation": regroup()} if consolidate else {}),
                        **walked,
                    }
                )
            else:
                record(
                    {
                        "step": step,
                        "epoch": epoch,
                        "screen": round(score, 4),
                        "kinds": _kinds(kinds),
                        **guarded,
                        **thermo,
                    }
                )
            if frozen >= STALL_EPOCHS:
                stop = "stalled"
                break
            if score == 1.0:
                stop = "passed"
                break
            if score > best or melting:  # the melt's screen is its best (D18)
                best, since = score, 0
                if restore_best:
                    net.save(str(step_best))
            else:
                since += 1
                if since >= settings.patience:
                    stop = "plateau"
                    break
            checkpoint(step, epoch)
        if keep_machine and stop == "plateau" and machine_best:
            net = load_network(str(machine_kept))
            restored = screen(net, encoding, screens)[0]
            record(
                {
                    "step": step,
                    "epoch": epoch,
                    "restore_machine": round(machine_best[0], 3),
                    "restored": round(restored, 4),
                }
            )
        if restore_best and stop == "plateau":
            net = load_network(str(step_best))
            restored = screen(net, encoding, screens)[0]
            grouped = {"consolidation": regroup()} if consolidate else {}
            record({"step": step, "epoch": epoch, "restore": round(best, 4), "restored": round(restored, 4), **grouped})
        net.save(str(out / "network.json"))
        net.save(str(out / f"step-{step}.network.json"))  # every step's kept, to inspect what a run had
        # each step's certified answer (the quotient) certified (under a second at one column a window); the run stops at
        # its first certified step: training past it can destroy the network (E8b; this probe's
        # first runs, both certified networks lost within 3 steps)
        certificate = wc.certify(QueryMachine(net, encoding, 0), task)
        record(
            {
                "step": step,
                "end": stop,
                "epochs": epoch,
                **({"forks": forks} if fork is not None else {}),
                **({"rollbacks": step_rollbacks} if guarding is not None else {}),
                **diagnose(net, encoding, settings),
                "long": long_accuracy(net, encoding, settings),
                "certified": certificate.certified,
                "minimised": len(certificate.minimised),
                "machine": machine_summary(net, encoding),
            }
        )
        if certificate.certified:
            net.save(str(out / f"certified-step-{step}.network.json"))
            break
        sample = [screens[n] for n in read_at]
        distinct = distinct_states(net, sample) if warm and warm_up is not None else []
        if warm and warm_up is not None:
            record({"step": step, "distinct": distinct})
        if warm and warm_up is not None and warm_up.ends(step, first, max(best, score), distinct, counts_snapped):
            # WarmUp's end (D14-D16): the codes from the states seen, each factor's T_hot, the profile from here
            warm = False
            warming()
            rng_codes = np.random.default_rng([seed, step])
            made = codes_from_states(net, sample, rng_codes, warm_up.iterations, warm_up.hot)
            hot = [m.get("hot", 1.0) for m in made]
            melting_points, last_splits = handover_window(made, warm_up.window)
            anneal_from = spent
            cooling = Cooling.start(hot) if warm_up.shape == "adaptive" else None
            fixed = step if crucible else 0
            record({"step": step, "warm_up_end": made, **({"crucible": fixed} if crucible else {})})
            net.save(str(out / "network.json"))
        if stop in ("deadline", "spent", "stalled"):
            break
    log.close()
    _write_checkpoint(out, identity, ENDED, net)


def guard_epoch(
    path: Path, encoding: Target, step: int, settings: Settings = PROBE, generator: str = "guard-epoch"
) -> dict[str, Any]:
    """
    One epoch of a step's training from a saved network, unguarded and with the guard armed, on the
    same data: whether its certified answer's machine is right before, and for each its seconds, whether
    right after and the guard's updates and rollbacks (the guard workplan's budget probe).
    """
    task = encoding.task
    result: dict[str, Any] = {
        "network": str(path),
        "step": step,
        "right_before": cg.right(QueryMachine(load_network(str(path)), encoding, 0), task),
    }
    for arm in ("unguarded", "guarded"):
        net = load_network(str(path))
        rng = random.Random(f"{generator}-{step}")
        examples = [encoding.example(encoding.draw(n, rng)) for n in epoch_lengths(step, settings.epoch_examples)]
        guard = cg.Guard(lambda n: QueryMachine(n, encoding, 0), task) if arm == "guarded" else None
        start = time.perf_counter()
        if guard is not None:
            guard.arm(net)
        wr.train_epoch(net, examples, settings.batch_size, settings.rate, rng, guard)
        result[arm] = {
            "seconds": round(time.perf_counter() - start, 1),
            "right_after": cg.right(QueryMachine(net, encoding, 0), task),
            "updates": None if guard is None else guard.updates,
            "rollbacks": None if guard is None else guard.rollbacks,
        }
    return result


def readout_only(net: Any) -> wr.FactorRates:
    """
    The network's optimizer with every index at a rate of 0 but the answers' path: the output layer,
    and with branches the answers' whole branch (the window branches workplan), the layers after the
    last factor's branch's end. The rest keep their weights, so the state (each pooling reads the
    output layer's input, or its branch's end) is the saved network's, and only the readout of its
    answers learns.
    """
    ends = [i for i, layer in enumerate(net.layers) if isinstance(layer, BranchArrayLayer)]
    learning = range(ends[-1] + 1 if ends else len(net.layers) - 1, len(net.layers))
    return wr.FactorRates(net.optimizer, {i: 0.0 for i in range(len(net.layers) + len(net.parts)) if i not in learning})


def tune_readout(
    path: Path,
    encoding: Target,
    task_name: str,
    step: int,
    out: Path,
    epochs: int,
    settings: Settings = PROBE,
    generator: str = "readout-tune",
) -> list[dict[str, Any]]:
    """
    A saved network's output layer trained alone (readout_only) on step's lengths, an epoch at a time,
    until its certificate (the task's slot, the default snap ratio) passes or epochs are spent: a row
    an epoch (0: before), each with the certificate's counts, out/log.jsonl. The network is saved at
    the end (out/network.json) and, if certified, out/certified.network.json.
    """
    out.mkdir(parents=True, exist_ok=True)
    net = load_network(str(path))
    task, index = wt.task_named(task_name), slot(task_name)
    rows: list[dict[str, Any]] = []
    with (out / "log.jsonl").open("w") as log:
        for epoch in range(epochs + 1):
            if epoch:
                rng = random.Random(f"{generator}-{step}-{epoch}")
                examples = [
                    encoding.example(encoding.draw(n, rng)) for n in epoch_lengths(step, settings.epoch_examples)
                ]
                inner, net.optimizer = net.optimizer, readout_only(net)
                try:
                    wr.train_epoch(net, examples, settings.batch_size, settings.rate, rng)
                finally:
                    net.optimizer = inner
            certificate = wc.certify(QueryMachine(net, encoding, index), task)
            rows.append(
                {
                    "epoch": epoch,
                    "certified": certificate.certified,
                    "wrong_outputs": certificate.wrong_outputs,
                    "wrong_states": certificate.wrong_states,
                    "minimised": len(certificate.minimised),
                    "lead": round(certificate.lead.smallest, 4),
                    "lead_short": certificate.lead.short,
                }
            )
            log.write(json.dumps(rows[-1]) + "\n")
            log.flush()
            if certificate.certified:
                net.save(str(out / "certified.network.json"))
                break
    net.save(str(out / "network.json"))
    return rows


def separability(
    path: Path, encoding: Target, task_name: str, steps: int = 60_000, passes: int = 20_000, seed: int = 0
) -> dict[str, Any]:
    """
    Whether a saved network's answers (its task's slot) are linearly separable over the features its
    output layer reads (the window branches workplan's check): the features for every reachable code
    and every window, an unregularised softmax readout fitted to them all (full batch, Adam, steps
    steps, its fewest wrong at each 5,000), then a multiclass perceptron (0 errors in finitely many
    passes iff separable) for at most passes passes; for each pair the fit leaves wrong, the distance
    to the nearest feature with another answer and with its own.
    """
    net = load_network(str(path))
    machine, task = QueryMachine(net, encoding, slot(task_name)), wt.task_named(task_name).machine
    windows = wc.all_windows(1, task.tokens)
    want_labels, want_state = task.windows(windows)
    start, _ = machine.start()
    phi = {start: task.start}
    queue = [start]
    for q in queue:  # every reachable code, breadth first, its target state
        states = machine.step(q, windows).states
        for window in range(len(windows)):
            r = tuple(int(c) for c in states[window])
            if r not in phi:
                phi[r] = int(want_state[phi[q]][window])
                queue.append(r)
    codes = list(phi)
    queries = np.array(encoding.query_tokens, dtype=np.int64)
    full = np.concatenate([windows, np.repeat(queries[np.newaxis, :], len(windows), axis=0)], axis=1)
    books = machine.inner.E
    vectors = np.array([np.concatenate([books[g][i] for g, i in enumerate(q)]) for q in codes])
    net._window_forward(
        net.backend.matrix(np.tile(full, (len(codes), 1)).astype(np.float64).tolist()),
        net.backend.matrix(np.repeat(vectors, len(windows), axis=0).tolist()),
    )
    features = np.array(net._activations[-2].tolist(), dtype=np.float64)
    width = features.shape[1] // full.shape[1]
    index = slot(task_name)
    feats = features[:, index * width : (index + 1) * width]
    y = np.array([int(want_labels[phi[q]][w][0]) for q in codes for w in range(len(windows))])
    outputs = np.array(net._activations[-1].tolist())
    k = net.class_count
    network_wrong = int((outputs[:, index * k : (index + 1) * k].argmax(axis=1) != y).sum())
    design: npt.NDArray[np.float64] = np.hstack([feats, np.ones((len(feats), 1))])
    onehot = np.eye(k)[y]
    weights: npt.NDArray[np.float64] = np.zeros((design.shape[1], k))
    m, v = np.zeros_like(weights), np.zeros_like(weights)
    fit: list[int] = []
    for step in range(1, steps + 1):
        z = design @ weights
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        g = design.T @ (p - onehot) / len(y)
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        weights -= 0.01 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-12)
        if step % 5000 == 0:
            fit.append(int(((design @ weights).argmax(axis=1) != y).sum()))
            if fit[-1] == 0:
                break
    rng = np.random.default_rng(seed)
    u: npt.NDArray[np.float64] = np.zeros((design.shape[1], k))
    fewest, used = len(y), 0
    for used in range(1, passes + 1):
        errors = 0
        for i in rng.permutation(len(y)):
            guess = int(np.argmax(design[i] @ u))
            if guess != y[i]:
                u[:, y[i]] += design[i]
                u[:, guess] -= design[i]
                errors += 1
        fewest = min(fewest, errors)
        if errors == 0:
            break
    wrong: list[dict[str, Any]] = []
    for i in np.flatnonzero((design @ weights).argmax(axis=1) != y):
        distances = np.sqrt(((feats - feats[i]) ** 2).sum(axis=1))
        own = np.sort(distances[y == y[i]])
        other = distances[y != y[i]]
        q = codes[i // len(windows)]
        wrong.append(
            {
                "code": list(q),
                "state": phi[q],
                "window": int(i % len(windows)),
                "answer": int(y[i]),
                "other": round(float(other.min()), 4),
                "own": round(float(own[1]), 4) if len(own) > 1 else None,
            }
        )
    return {
        "network": str(path),
        "codes": len(codes),
        "states": len(set(phi.values())),
        "pairs": len(y),
        "features": width,
        "network_wrong": network_wrong,
        "fit_wrong": fit,
        "perceptron": {"passes": used, "fewest": fewest, "separable": fewest == 0},
        "wrong": wrong,
    }


class QueryMachine:
    """
    A family network as the certificate's WindowMachine over one column a window (stage 3): each
    window c is run as the network's window [c, queries...], and one slot's label read, slot 0 the
    quotient (against division's machine) or a machine's own output, slot i the i-th query's operation
    (the sum against the modular sum's), or with slot None every slot's (A's groups, read on all
    the network's outputs). The states, their snaps and margins are the network's own, as NetworkMachine's.
    """

    def __init__(self, network: Any, encoding: Target, slot: int | None) -> None:
        assert network.columns == encoding.width, "the network's window is the encoding's"
        self.inner = wc.NetworkMachine(network)
        self.encoding, self.slot = encoding, slot

    @property
    def columns(self) -> int:
        return 1

    @property
    def codes(self) -> list[int]:
        return self.inner.codes

    def start(self) -> tuple[wc.State, list[wc.FloatArray]]:
        return self.inner.start()

    def step(self, state: wc.State, windows: wc.IntArray) -> wc.Steps:
        return self.step_states([state], windows)

    def step_states(self, states: Sequence[wc.State], windows: wc.IntArray) -> wc.Steps:
        queries = np.array(self.encoding.query_tokens, dtype=np.int64)
        full = np.concatenate([windows, np.repeat(queries[np.newaxis, :], len(windows), axis=0)], axis=1)
        steps = self.inner.step_states(states, full)
        s = self.slot
        if s is None:
            return steps
        return wc.Steps(steps.digits[:, s : s + 1], steps.leads[:, s : s + 1], steps.states, steps.distances)


SLOTS = {"modular-sum": 1}  # the task each slot is certified against; any other task, slot 0


def slot(task_name: str) -> int:
    """The window slot a task is certified and long-tested on: the modular sum's 1, any other task's 0."""
    return SLOTS.get(task_name.rsplit("-", 1)[0], 0)


def certify_file(path: Path, encoding: Target, task_name: str, snap_ratio: float = wc.SNAP_RATIO) -> dict[str, Any]:
    """A saved family network's certificate, one slot against its task's machine, as certify_file's JSON."""
    net = load_network(str(path))
    index = slot(task_name)
    certificate = wc.certify(QueryMachine(net, encoding, index), wt.task_named(task_name), snap_ratio)
    return {
        "network": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "slot": index,
        "snap_ratio": snap_ratio,
        **certificate.to_json(),
    }


class QueryModel:
    """
    A family network as a task's catalogue reads it at a length n (the harness's WindowModel at k
    1, so every alignment 0): each case's columns and the final zero column as the encoding's
    windows, one slot's labels read, in runs of at most the harness's CHUNK_TOKENS tokens.
    """

    def __init__(self, network: Any, encoding: Target, task: wt.Task, slot: int) -> None:
        assert network.columns == encoding.width, "the network's window is the encoding's"
        self.network, self.encoding, self.task, self.slot = network, encoding, task, slot

    def predict(self, n: int) -> wt.Predict:
        width = self.encoding.width
        rows = max(1, wr.CHUNK_TOKENS // ((n + 1) * width))

        def labels(cases: Sequence[Any]) -> list[wt.Labels]:
            result: list[wt.Labels] = []
            for j in range(0, len(cases), rows):
                chunk = [self.encoding.example((*self.task.column_tokens(c, n), 0)) for c in cases[j : j + rows]]
                predictions = self.network.classify_rows(self.network.prepare_dataset(chunk))
                result.extend(tuple(prediction[self.slot :: width]) for prediction in predictions)
            return result

        return labels


def long_test(
    path: Path,
    encoding: Target,
    task_name: str,
    lengths: Sequence[int] = wr.LONG_LENGTHS,
    properties: Sequence[str] = LONG_PROPERTIES,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    A saved family network's long tests (the division workplan's D4), one slot against its task's
    catalogue at each length, sized as the harness's long_test, as its JSON (so
    window_recurrent_study.long_test_failures reads it), with the network's sha256 and the slot.
    """
    task = wt.task_named(task_name)
    index = slot(task_name)
    catalogue = task.catalogue(ad.SAMPLED, ad.RELATION_SAMPLED, ad.PER_STRATUM, doubling=True)
    tested = [p for p in wr.test_properties(task, catalogue) if p.id in properties]
    model = QueryModel(load_network(str(path)), encoding, task, index)
    results: dict[str, Any] = {}
    for n in lengths:
        start = time.perf_counter()
        checked = task.evaluate(model.predict(n), n, tested)
        if log is not None:
            for result in checked:
                failure = "" if result.holds else f", e.g. {result.failures[0]}"
                log(f"n {n}, {result.id}: {result.passed}/{result.cases}{failure}")
        results[str(n)] = {
            "properties": {r.id: task.result_json(r) for r in checked},
            "seconds": time.perf_counter() - start,
        }
    return {
        "network": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "task": task_name,
        "slot": index,
        "lengths": results,
    }
