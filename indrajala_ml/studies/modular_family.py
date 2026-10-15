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

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from indrajala_ml.data import addition_data as ad
from indrajala_ml.data import modular_data as md
from indrajala_ml.model.networks.window_recurrent_array_network import WindowRecurrentRustArrayNetwork
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import BEFORE_STATE, AttentionPooling, Codes, Factor, WindowRecurrence
from indrajala_ml.studies import window_diagnostics as wd
from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_recurrent_study as wr
from indrajala_ml.studies import window_tasks as wt

BASE = md.BASE
COLUMNS = BASE * BASE  # a column's tokens, 3 a + b

# the operations a query token asks for, after the column's own quotient
OPERATIONS: dict[str, Callable[[int, int, int], int]] = {
    "sum": lambda A, B, p: (A + B) % p,
    "difference": lambda A, B, p: (A - B) % p,
}
FAMILY = ("sum", "difference")
SUM = ("sum",)  # the family without the difference (the division workplan's stage 3 option)
QUOTIENT: tuple[str, ...] = ()

Example = wr.Example

LONG_PROPERTIES = ("U1", "S1")  # the long tests' (the division workplan's D4)


@dataclass(frozen=True)
class Encoding:
    """A window's layout: the column, then a query token per operation, each its own answer."""

    p: int
    queries: tuple[str, ...]

    @property
    def width(self) -> int:
        return 1 + len(self.queries)

    @property
    def tokens(self) -> int:
        return COLUMNS + len(OPERATIONS)

    @property
    def classes(self) -> int:
        return self.p + 1  # residues and undefined

    def query_token(self, operation: str) -> int:
        return COLUMNS + list(OPERATIONS).index(operation)

    def states(self, columns: Sequence[int]) -> list[tuple[int, int]]:
        """(A, B) mod p after each column."""
        a = b = 0  # the prefixes mod p
        states: list[tuple[int, int]] = []
        for token in columns:
            a, b = (BASE * a + token // BASE) % self.p, (BASE * b + token % BASE) % self.p
            states.append((a, b))
        return states

    def example(self, columns: Sequence[int]) -> Example:
        """The windows' tokens and each token's answer."""
        tokens: list[float] = []
        labels: list[int] = []
        for token, (A, B) in zip(columns, self.states(columns), strict=True):
            tokens.append(float(token))
            labels.append(md.quotient(A, B, self.p))
            for operation in self.queries:
                tokens.append(float(self.query_token(operation)))
                labels.append(OPERATIONS[operation](A, B, self.p))
        return tuple(tokens), tuple(labels)


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
    Snapshot and fork on collapse (docs/collapse-fork-workplan.md): within a step, the network at the
    step's best screen score is kept; an epoch whose score falls below (1 - drop) of that peak is a
    collapse, and the run goes back to the snapshot, cools the step's rate by `cool` and takes its
    data from a new stream; at most `limit` forks a step.
    """

    drop: float = 0.4  # D1
    cool: float = 0.5  # D3
    limit: int = 4  # D4


def screen_score(net: Any, encoding: Encoding, screens: Mapping[int, Sequence[Example]]) -> float:
    """The share of every answer right over a step's screens (its training lengths)."""
    tallies = [accuracy(net, encoding, rows) for rows in screens.values()]
    correct = sum(c for t in tallies for c, _ in t.values())
    return correct / sum(a for t in tallies for _, a in t.values())


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


def trace(rows: Sequence[Mapping[str, Any]], drop: float = 0.5) -> list[str]:
    """
    A run's log.jsonl, a line per diagnosis: step, epoch (or the step's end), each factor's codes in
    use, the state's share of the pair, each answer's accuracy; marked where the codes in use fall
    below `drop` of the diagnosis before (a collapse: the division workplan's stage 4 trace).
    """
    lines: list[str] = []
    before: list[int] | None = None
    for row in rows:
        codes: list[int] = [int(factor["codes"]) for factor in row["factors"]]
        answers = " ".join(f"{name} {value:.2f}" for name, value in sorted(row["accuracy"].items()))
        end = f" END {row['end']} min {row.get('minimised')}" if "end" in row else ""
        mark = " << collapse" if before is not None and sum(codes) < drop * sum(before) else ""
        epoch = row.get("epoch", "-")
        share = row.get("state_share", 0.0)
        lines.append(f"step {row['step']:>3} epoch {epoch:>3} codes {codes} share {share:.2f} {answers}{end}{mark}")
        before = codes
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
        peak = 0.0
        if fork is not None:
            peak = screen_score(net, encoding, screens)
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
            score = screen_score(net, encoding, screens)
            if fork is not None:
                if score > peak:
                    peak = score
                    net.save(str(snapshot))
                elif score < (1 - fork.drop) * peak and forks < fork.limit:
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
                            "peak": round(peak, 4),
                            "restored": round(restored, 4),
                            "rate": rate,
                        }
                    )
                    continue
            if epoch % settings.diagnose_every == 0:
                record({"step": step, "epoch": epoch, "screen": round(score, 4), **diagnose(net, encoding, settings)})
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
