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
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from indrajala_ml.data import addition_data as ad
from indrajala_ml.data import modular_data as md
from indrajala_ml.model.networks.window_recurrent_array_network import WindowRecurrentRustArrayNetwork
from indrajala_ml.model.persistence.format2 import network_to_json
from indrajala_ml.model.persistence.load_network import load_network, network_from_state
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import BEFORE_STATE, AttentionPooling, Codes, Factor, WindowRecurrence
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


def network(encoding: Target, factors: Sequence[tuple[int, int]], seed: int, reads: str = BEFORE_STATE) -> Any:
    """The transformer window (W11) over the encoding's tokens, a factored state (D3), Adam; the
    poolings reading before the first StateInput (D10) or, with reads OUTPUT, the window's output."""
    recurrence = WindowRecurrence(
        tuple(Factor(s, Codes(m)) for s, m in factors), False, pooling=AttentionPooling(reads)
    )
    specs = wr.transformer_window(encoding.tokens, encoding.classes)
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
        if "factors" not in row:  # an epoch's screen score alone
            continue
        codes = _codes(row)
        answers = " ".join(f"{name} {value:.3f}" for name, value in sorted(row["accuracy"].items()))
        end = f" END {row['end']} min {row.get('minimised')}" if "end" in row else ""
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


def consolidation(net: Any, encoding: Target, d: int) -> tuple[dict[tuple[int, ...], tuple[int, ...]], int, int]:
    """
    A's table from the network alone (the workplan's D1): its reachable joint codes grouped by
    d-equivalence over every slot's outputs, each duplicate to its group's representative (the code
    its shortest word reaches; machine_reading.groups); with the codes reached and the groups.
    """
    walk = mr.walk(QueryMachine(net, encoding, None), encoding.task.machine.tokens)
    found = mr.groups(walk, d)
    return mr.consolidation(walk, found), len(walk.codes), len(found)


CHECKPOINT = "checkpoint.json"


@dataclass
class Place:
    """
    Where a run is, as its checkpoint keeps it (run's resume): the step and the last epoch trained
    in it (0: the step just begun), the epochs trained in all, the step's best screen and epochs
    since, its rate, forks and the fork's peak, the last screen's answer kinds, the step's undone
    updates and the guard's frozen epochs; the seconds trained, and the log's length in bytes; and
    A's table, (duplicate, representative) pairs (training state, not saved with the network).
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
    }


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
    fork's or a restore's loaded network has it too).

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
        net = network(encoding, factors, seed, reads)
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
        )
        _write_checkpoint(out, identity, place, net)

    def regroup() -> dict[str, int]:
        # A's table read from the network as it stands and held until the next; the row's record of it
        nonlocal table
        table, codes, groups = consolidation(net, encoding, d)
        net.consolidate(table)
        return {"codes": codes, "groups": groups, "duplicates": len(table)}

    for step in range(place.step if place is not None else first, cap + 1):
        lengths = list(range(1, step + 1))
        screens = {n: _windows(encoding, n, settings.screen, f"family-screen-{step}-{n}")[0] for n in lengths}
        stop, epoch = "budget", 0
        if place is not None and place.step == step:  # resumed within this step
            best, since, epoch = place.best, place.since, place.epoch
            rate, forks, peak = place.rate, place.forks, place.peak
            last_kinds, step_rollbacks = place.last_kinds, place.step_rollbacks
        else:
            best, since = -1.0, 0
            rate, forks = settings.rate, 0
            start_score, start_kinds = screen(
                net, encoding, screens
            )  # the step before's network on this step's lengths
            record({"step": step, "epoch": 0, "screen": round(start_score, 4), "kinds": _kinds(start_kinds)})
            last_kinds, step_rollbacks = start_kinds, 0
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
                for n in epoch_lengths(step, settings.epoch_examples, long_tail, cap)
            ]
            if commitment_from is not None:
                net.set_commitments([0.0 if step < commitment_from else c for c in commitments])
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
                            **grouped,
                        }
                    )
                    checkpoint(step, epoch)
                    continue
            if epoch % settings.diagnose_every == 0:
                record(
                    {
                        "step": step,
                        "epoch": epoch,
                        "screen": round(score, 4),
                        "kinds": _kinds(kinds),
                        **guarded,
                        **diagnose(net, encoding, settings),
                        **({"consolidation": regroup()} if consolidate else {}),
                    }
                )
            else:
                record({"step": step, "epoch": epoch, "screen": round(score, 4), "kinds": _kinds(kinds), **guarded})
            if frozen >= STALL_EPOCHS:
                stop = "stalled"
                break
            if score == 1.0:
                stop = "passed"
                break
            if score > best:
                best, since = score, 0
                if restore_best:
                    net.save(str(step_best))
            else:
                since += 1
                if since >= settings.patience:
                    stop = "plateau"
                    break
            checkpoint(step, epoch)
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
            }
        )
        if certificate.certified:
            net.save(str(out / f"certified-step-{step}.network.json"))
            break
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
