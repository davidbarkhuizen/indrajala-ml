"""
The modular family (docs/modular-family-workplan.md): one window-recurrent network answering
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
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from indrajala_ml.data import addition_data as ad
from indrajala_ml.data import modular_data as md
from indrajala_ml.model.networks.window_recurrent_array_network import WindowRecurrentRustArrayNetwork
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import BEFORE_STATE, AttentionPooling, Codes, Factor, WindowRecurrence
from indrajala_ml.studies import inference
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


def columns(n: int, rng: random.Random) -> list[int]:
    """n uniform columns, then the final zero column (the window studies' final zero window, at
    k 1: its answer needs the state the data left, so even one column trains the state)."""
    return [rng.randrange(COLUMNS) for _ in range(n)] + [0]


def network(encoding: Encoding, factors: Sequence[tuple[int, int]], seed: int, reads: str = BEFORE_STATE) -> Any:
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


def screen_score(net: Any, encoding: Encoding, screens: Mapping[int, Sequence[Example]]) -> float:
    """The share of every answer right over a step's screens (its training lengths)."""
    return screen(net, encoding, screens)[0]


def screen(net: Any, encoding: Encoding, screens: Mapping[int, Sequence[Example]]) -> tuple[float, dict[str, float]]:
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


def _kinds(kinds: Mapping[str, float]) -> dict[str, float]:
    return {name: round(share, 4) for name, share in kinds.items()}


def _windows(encoding: Encoding, n: int, cases: int, seed: str) -> tuple[list[Example], list[list[int]]]:
    rng = random.Random(seed)
    drawn = [columns(n, rng) for _ in range(cases)]
    return [encoding.example(c) for c in drawn], drawn


def accuracy(net: Any, encoding: Encoding, rows: Sequence[Example]) -> dict[str, list[int]]:
    """Each answer's [correct, all], the quotient by class (0, undefined, nonzero)."""
    tally: dict[str, list[int]] = {}
    predictions: list[Sequence[int]] = net.classify_rows(net.prepare_dataset(rows))
    for (_, labels), predicted in zip(rows, predictions, strict=True):
        for i, (label, guess) in enumerate(zip(labels, predicted, strict=True)):
            slot = i % encoding.width
            if slot == 0:
                name = (
                    "quotient-0" if label == 0 else "quotient-undefined" if label == encoding.p else "quotient-nonzero"
                )
            else:
                name = encoding.queries[slot - 1]
            t = tally.setdefault(name, [0, 0])
            t[0] += int(label == guess)
            t[1] += 1
    return tally


def diagnose(net: Any, encoding: Encoding, settings: Settings) -> dict[str, Any]:
    """Each answer's accuracy and each factor's codes used and shares, at the diagnosis's lengths."""
    tally: dict[str, list[int]] = {}
    samples: list[tuple[tuple[int, ...], int, int]] = []  # codes, A, B per window
    for n in settings.lengths:
        rows, drawn = _windows(encoding, n, settings.cases, f"family-diagnose-{n}")
        for name, (c, a) in accuracy(net, encoding, rows).items():
            t = tally.setdefault(name, [0, 0])
            t[0] += c
            t[1] += a
        codes: list[list[tuple[int, ...]]] = net.state_codes(net.backend.matrix([tokens for tokens, _ in rows]))
        for states, cols in zip(codes, drawn, strict=True):
            for w, (A, B) in enumerate(encoding.states(cols)):
                samples.append((states[w + 1], A, B))
    p = encoding.p
    quantities: dict[str, Callable[[int, int], int]] = {
        "A": lambda A, B: A,
        "B": lambda A, B: B,
        "S": lambda A, B: (A + B) % p,
        "D": lambda A, B: (A - B) % p,
        "A=0": lambda A, B: int(A == 0),
        "B=0": lambda A, B: int(B == 0),
    }
    factors = [
        {
            "codes": len({c[g] for c, _, _ in samples}),
            "shares": {q: round(wd.share([(c[g], f(A, B)) for c, A, B in samples]), 3) for q, f in quantities.items()},
        }
        for g in range(len(samples[0][0]))
    ]
    return {
        "accuracy": {name: round(c / a, 3) for name, (c, a) in sorted(tally.items())},
        "factors": factors,
        "state_share": round(wd.share([(c, (A, B)) for c, A, B in samples]), 3),
    }


def long_accuracy(net: Any, encoding: Encoding, settings: Settings) -> dict[str, float]:
    """The quotient's accuracy over every window at each long length."""
    result: dict[str, float] = {}
    for n in settings.long:
        rows, _ = _windows(encoding, n, settings.cases, f"family-long-{n}")
        tally = accuracy(net, encoding, rows)
        q = [tally[k] for k in tally if k.startswith("quotient")]
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
    line per fork. An epoch's screen score alone (a row without codes) isn't shown.
    """
    lines: list[str] = []
    before: list[int] | None = None
    for row in rows:
        if "fork" in row:
            lines.append(
                f"step {row['step']:>3} epoch {row['epoch']:>3} FORK {row['fork']}: screen {row['screen']:.2f} "
                f"peak {row['peak']:.2f} restored {row['restored']:.2f} rate {row['rate']:.2g}"
            )
            continue
        if "factors" not in row:  # an epoch's screen score alone
            continue
        codes = _codes(row)
        answers = " ".join(f"{name} {value:.2f}" for name, value in sorted(row["accuracy"].items()))
        end = f" END {row['end']} min {row.get('minimised')}" if "end" in row else ""
        mark = " << collapse" if _collapsed(before, codes, drop) else ""
        epoch = row.get("epoch", "-")
        share = row.get("state_share", 0.0)
        lines.append(f"step {row['step']:>3} epoch {epoch:>3} codes {codes} share {share:.2f} {answers}{end}{mark}")
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
        round(min(best.get("accuracy", {"-": 0.0}).values()), 2),
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
                f"worst answer at best {run.peak:.2f} (step {run.peak_step}), collapses {run.collapses}, "
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


def exclusion_bound(p: int, factors: Sequence[tuple[int, int]]) -> int:
    """`m + q - 1` (experiment-design.md, section 1): the joint codes plus division's p * p states, less one."""
    return math.prod(m for _, m in factors) + p * p - 1


def run(
    encoding: Encoding,
    factors: Sequence[tuple[int, int]],
    seed: int,
    budget_seconds: float,
    out: Path,
    settings: Settings = PROBE,
    cap: int | None = None,
    reads: str = BEFORE_STATE,
    budget_epochs: int | None = None,
    fork: Fork | None = None,
) -> None:
    """
    One probe run: the curriculum until the budget (wall-clock seconds and, if given, epochs
    trained in all, whichever is spent first: the epochs don't depend on the machine or its load)
    or the cap (by default the cell's exclusion bound), a line of JSON per diagnosis to
    out/log.jsonl, the network saved at each step's end to out/network.json. With `fork`, a collapse
    within a step goes back to the step's best network and forks (Fork; a log line each).
    """
    if cap is None:
        cap = exclusion_bound(encoding.p, factors)
    out.mkdir(parents=True, exist_ok=True)
    net = network(encoding, factors, seed, reads)
    start = time.perf_counter()
    log = (out / "log.jsonl").open("a")

    def record(row: dict[str, Any]) -> None:
        row["seconds"] = round(time.perf_counter() - start, 1)
        log.write(json.dumps(row) + "\n")
        log.flush()

    record({"step": 0, "epoch": 0, **diagnose(net, encoding, settings)})
    spent = 0  # epochs trained in all
    for step in range(1, cap + 1):
        lengths = list(range(1, step + 1))
        screens = {n: _windows(encoding, n, settings.screen, f"family-screen-{step}-{n}")[0] for n in lengths}
        best, since, stop, epoch = -1.0, 0, "budget", 0
        rate, forks, snapshot = settings.rate, 0, out / "snapshot.network.json"
        start_score, start_kinds = screen(net, encoding, screens)  # the step before's network on this step's lengths
        record({"step": step, "epoch": 0, "screen": round(start_score, 4), "kinds": _kinds(start_kinds)})
        peak = 0.0
        if fork is not None:
            peak = fork.trigger.value(start_score, start_kinds)
            net.save(str(snapshot))
        for epoch in range(1, settings.max_epochs + 1):
            if time.perf_counter() - start > budget_seconds:
                stop = "deadline"
                break
            if budget_epochs is not None and spent >= budget_epochs:
                stop = "spent"
                break
            rng = random.Random(f"family-train-{seed}-{step}-{epoch}" + (f"-fork{forks}" if forks else ""))
            examples = [
                encoding.example(columns(lengths[i % len(lengths)], rng)) for i in range(settings.epoch_examples)
            ]
            wr.train_epoch(net, examples, settings.batch_size, rate, rng)
            spent += 1
            score, kinds = screen(net, encoding, screens)
            if fork is not None:
                value = fork.trigger.value(score, kinds)
                if value > peak:
                    peak = value
                    net.save(str(snapshot))
                elif fork.trigger.fires(value, peak) and forks < fork.limit:
                    forks += 1
                    rate *= fork.cool
                    net = load_network(str(snapshot))
                    restored = screen_score(net, encoding, screens)
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
                        }
                    )
                    continue
            if epoch % settings.diagnose_every == 0:
                record(
                    {
                        "step": step,
                        "epoch": epoch,
                        "screen": round(score, 4),
                        "kinds": _kinds(kinds),
                        **diagnose(net, encoding, settings),
                    }
                )
            else:
                record({"step": step, "epoch": epoch, "screen": round(score, 4), "kinds": _kinds(kinds)})
            if score == 1.0:
                stop = "passed"
                break
            if score > best:
                best, since = score, 0
            else:
                since += 1
                if since >= settings.patience:
                    stop = "plateau"
                    break
        net.save(str(out / "network.json"))
        net.save(str(out / f"step-{step}.network.json"))  # every step's kept, to inspect what a run had
        # each step's quotient certified (under a second at one column a window); the run stops at
        # its first certified step: training past it can destroy the network (E8b; this probe's
        # first runs, both certified networks lost within 3 steps)
        certificate = wc.certify(QueryMachine(net, encoding, 0), wt.task_named(f"modular-division-{encoding.p}"))
        record(
            {
                "step": step,
                "end": stop,
                "epochs": epoch,
                **({"forks": forks} if fork is not None else {}),
                **diagnose(net, encoding, settings),
                "long": long_accuracy(net, encoding, settings),
                "certified": certificate.certified,
                "minimised": len(certificate.minimised),
            }
        )
        if certificate.certified:
            net.save(str(out / f"certified-step-{step}.network.json"))
            break
        if stop in ("deadline", "spent"):
            break
    log.close()


class QueryMachine:
    """
    A family network as the certificate's WindowMachine over one column a window (stage 3): each
    window c is run as the network's window [c, queries...], and one slot's label read, slot 0 the
    quotient (against division's machine), slot i the i-th query's operation (the sum against the
    modular sum's). The states, their snaps and margins are the network's own, as NetworkMachine's.
    """

    def __init__(self, network: Any, encoding: Encoding, slot: int) -> None:
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
        queries = np.array([self.encoding.query_token(q) for q in self.encoding.queries], dtype=np.int64)
        full = np.concatenate([windows, np.repeat(queries[np.newaxis, :], len(windows), axis=0)], axis=1)
        steps = self.inner.step(state, full)
        s = self.slot
        return wc.Steps(steps.digits[:, s : s + 1], steps.leads[:, s : s + 1], steps.states, steps.distances)


SLOTS = {"modular-division": 0, "modular-sum": 1}  # the task each slot is certified against


def certify_file(path: Path, encoding: Encoding, task_name: str, snap_ratio: float = wc.SNAP_RATIO) -> dict[str, Any]:
    """A saved family network's certificate, one slot against its task's machine, as certify_file's JSON."""
    net = load_network(str(path))
    slot = SLOTS[task_name.rsplit("-", 1)[0]]
    certificate = wc.certify(QueryMachine(net, encoding, slot), wt.task_named(task_name), snap_ratio)
    return {
        "network": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "slot": slot,
        "snap_ratio": snap_ratio,
        **certificate.to_json(),
    }


class QueryModel:
    """
    A family network as a task's catalogue reads it at a length n (the harness's WindowModel at k
    1, so every alignment 0): each case's columns and the final zero column as the encoding's
    windows, one slot's labels read, in runs of at most the harness's CHUNK_TOKENS tokens.
    """

    def __init__(self, network: Any, encoding: Encoding, task: wt.Task, slot: int) -> None:
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
    encoding: Encoding,
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
    slot = SLOTS[task_name.rsplit("-", 1)[0]]
    catalogue = task.catalogue(ad.SAMPLED, ad.RELATION_SAMPLED, ad.PER_STRATUM, doubling=True)
    tested = [p for p in wr.test_properties(task, catalogue) if p.id in properties]
    model = QueryModel(load_network(str(path)), encoding, task, slot)
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
        "slot": slot,
        "lengths": results,
    }
