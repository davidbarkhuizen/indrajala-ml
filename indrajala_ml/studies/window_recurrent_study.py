"""
The window-recurrent adder's harness (docs/window-recurrent-adder-workplan.md, stage 8;
scripts/window_recurrent_study.py), for any of window_tasks' tasks (the claims workplan's stage 5):
addition, the matrix's, or modular division.

A run is one task, arm, window size k, state width s, code count m and seed: W11's window (d 16, 3
causal attention blocks of 2 heads, each read the state through a StateInput, FFN width 32;
embedding the task's tokens, an output of its classes) beside a WindowRecurrence(s, Codes(m)), on
the Rust backend, under Adam. m None is a continuous state (WindowRecurrence(s, None), the claims
workplan's E2): its run has no codes to count (codes_used None) and its key says "continuous". A
run's window may instead be one of E6's baselines (WINDOWS: "linear", no blocks; "ffn", the blocks
without attention), its key and the report's cell naming it.

A case of n columns is read at an alignment a (W6-i): a zero columns, the n column tokens in the
task's reading order (addition's candidate 1's, 9 a_t + 3 b_t + c_t, least significant first), and
zero columns up to whole windows, then one window more (W9): ceil((n + a) / k) + 1 windows, a's
data windows ceil((n + a) / k). Its labels are the task's machine's outputs over those tokens (for
addition, the sum's digits from column a, the zero columns below labelled 0). In training a is
uniform in 0..k - 1; in a check it is the case's own, from the task's hash of the case (W23), the
same for every model.

The curriculum (W6) is steps of data window counts: step j trains on counts 1..j together
(cumulative), warm-started, the network and its Adam carried (W25); the all-at-once arm (W16) is
one step of counts 1..steps. An epoch draws an equal share of examples per count, each count's n
uniform among the lengths that fill that many windows at its alignment, its triple from D7's
mixture, held out (D8) from n = 3 (W24); its batches each of one window count, their order
shuffled (W26). A step's check is the catalogue at each count's longest length, n = w k (W24),
without B3 below n = 3: screened each epoch, the step stopping at the first epoch whose screen and
then whole check pass, at a plateau or at the budget, as the addition study's runs (W27).

After every step the test pass (W17): the catalogue without B3 at each test length; the break
length, the largest test length at which it and every shorter one pass (0 when the first fails),
and the window ratio, the break's data windows at a = 0 over the step's most (W10); M3's L* at each
test length; and the codes the states used, over every check length's first B1 cases (W4-i, W28).
With tests_past_break off (W12) the pass ends at its first failing length, tested whole: the break
and the ratio are the same, the codes used are counted over the lengths tested, and the lengths
past it are left to the saved network.
Every step records its examples and epochs (D10) and its seconds per epoch.

A run's steps are saved as they finish, each with its network (format 2, Adam's moments with it),
so a stopped run resumes at its next step; its result is written once its last step is done.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import multiprocessing
import os
import random
import socket
import subprocess
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from indrajala_ml.data import addition_data as ad
from indrajala_ml.model.networks.window_recurrent_array_network import WindowRecurrentRustArrayNetwork
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    Embedding,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
    StateInput,
)
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import Codes, Factor, WindowRecurrence
from indrajala_ml.studies import inference
from indrajala_ml.studies import window_tasks as wt
from indrajala_ml.studies.common import crate_hash, table

D, BLOCKS, HEADS, FFN = 16, 3, 2, 32  # W11
HOLD_OUT_FROM = 3  # W24: below it every case is trained on, and the held-out property (B3) isn't checked
RETRIES = 1
WORKS = 16  # W10: a model works when its break length is at least this
ARMS = ("curriculum", "all-at-once")
CHUNK_TOKENS = 1 << 20  # tokens a forward pass at most: memory bounded at any length (1,024 rows at 1,024 columns)

Example = tuple[tuple[float, ...], tuple[int, ...]]


def window_specs(task: wt.Task) -> list[LayerSpec]:
    """W11's window for a task: the state entering before each attention block (W18)."""
    blocks: list[LayerSpec] = []
    for _ in range(BLOCKS):
        blocks += [
            StateInput(),
            Residual((LayerNorm(), Attention(heads=HEADS, causal=True))),
            Residual((LayerNorm(), Dense(FFN, activation="relu"), Dense(D, activation="linear", bias=True))),
        ]
    return [
        Embedding(task.machine.tokens, D),
        Position(),
        *blocks,
        LayerNorm(),
        Dense(task.machine.classes, output=True, activation="softmax", loss="cross_entropy"),
    ]


def _output(task: wt.Task) -> list[LayerSpec]:
    return [LayerNorm(), Dense(task.machine.classes, output=True, activation="softmax", loss="cross_entropy")]


def linear_window_specs(task: wt.Task) -> list[LayerSpec]:
    """
    E6's baseline (the claims workplan, D21): no blocks, the state added to the embedded tokens and
    one linear residual (a StateInput stands before a residual block), then the output layer; no
    nonlinearity but the LayerNorms, so the recurrence does the step's work.
    """
    return [
        Embedding(task.machine.tokens, D),
        Position(),
        StateInput(),
        Residual((LayerNorm(), Dense(D, activation="linear", bias=True))),
        *_output(task),
    ]


def ffn_window_specs(task: wt.Task) -> list[LayerSpec]:
    """E6's attention control (D21): W11's window without its attention, the state before each FFN."""
    blocks: list[LayerSpec] = []
    for _ in range(BLOCKS):
        blocks += [
            StateInput(),
            Residual((LayerNorm(), Dense(FFN, activation="relu"), Dense(D, activation="linear", bias=True))),
        ]
    return [Embedding(task.machine.tokens, D), Position(), *blocks, *_output(task)]


TRANSFORMER = "transformer"
WINDOWS: dict[str, Callable[[wt.Task], list[LayerSpec]]] = {
    TRANSFORMER: window_specs,
    "linear": linear_window_specs,
    "ffn": ffn_window_specs,
}


def build(task: wt.Task, k: int, s: int, m: int | None, seed: int, window: str = TRANSFORMER) -> Any:
    """
    A run's network: its window (WINDOWS: W11's transformer, or E6's), and m codes or a continuous
    state for m None (the claims workplan's E2).
    """
    codes = None if m is None else Codes(m)
    return WindowRecurrentRustArrayNetwork.randomized(
        (k,), WINDOWS[window](task), WindowRecurrence((Factor(s, codes),)), Adam(), seed=seed
    )


# the encoding


def data_windows(n: int, a: int, k: int) -> int:
    return -(-(n + a) // k)


def alignment(task: wt.Task, case: Any, n: int, k: int) -> int:
    """W23: a case's alignment, from the high half of its hash (the low half decides D8's tenth)."""
    return (task.hash(case, n) >> 32) % k


def example(task: wt.Task, case: Any, n: int, k: int, a: int) -> Example:
    """The case's columns at alignment a, through its final zero window, and the task's labels for them."""
    width = (data_windows(n, a, k) + 1) * k
    columns = (0,) * a + task.column_tokens(case, n)
    tokens = columns + (0,) * (width - len(columns))
    return tuple(float(t) for t in tokens), task.machine.run(tokens)


def lengths_for(windows: int, a: int, k: int) -> range:
    """The lengths n >= 1 that fill exactly this many data windows at alignment a."""
    return range(max(1, (windows - 1) * k - a + 1), windows * k - a + 1)


def training_examples(task: wt.Task, counts: Sequence[int], k: int, count: int, rng: random.Random) -> list[Example]:
    """count examples, an equal share per data window count, the task's three kinds in turn within each (D7)."""
    examples: list[Example] = []
    for i in range(count):
        windows = counts[i % len(counts)]
        a = rng.randrange(k)
        n = rng.choice(lengths_for(windows, a, k))
        case = task.training_case(i // len(counts), n, rng, n >= HOLD_OUT_FROM)
        examples.append(example(task, case, n, k, a))
    return examples


def _by_width(examples: Sequence[Example]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for i, (tokens, _labels) in enumerate(examples):
        groups.setdefault(len(tokens), []).append(i)
    return groups


def train_epoch(network: Any, examples: Sequence[Example], batch_size: int, rate: float, rng: random.Random) -> None:
    """Batches of one window count each (a network batch's windows are the same), in shuffled order."""
    batches: list[list[Example]] = []
    for indices in _by_width(examples).values():
        rng.shuffle(indices)
        batches += [[examples[i] for i in indices[j : j + batch_size]] for j in range(0, len(indices), batch_size)]
    rng.shuffle(batches)
    for batch in batches:
        network.learn_batch(rate, batch)


class WindowModel:
    """A window-recurrent network as its task's catalogue reads it at a length n."""

    def __init__(self, network: Any, k: int, task: wt.Task) -> None:
        self.network, self.k, self.task = network, k, task

    def _chunks(self, cases: Sequence[Any], n: int) -> Iterator[tuple[list[int], list[Example]]]:
        """
        The cases' indices and examples, a width at a time in runs of at most CHUNK_TOKENS tokens,
        each run's examples built only when it comes: memory bounded at any length.
        """
        alignments = [alignment(self.task, c, n, self.k) for c in cases]
        groups: dict[int, list[int]] = {}
        for i, a in enumerate(alignments):
            groups.setdefault(data_windows(n, a, self.k), []).append(i)
        for windows, indices in groups.items():
            rows = max(1, CHUNK_TOKENS // ((windows + 1) * self.k))
            for j in range(0, len(indices), rows):
                chunk = indices[j : j + rows]
                yield chunk, [example(self.task, cases[i], n, self.k, alignments[i]) for i in chunk]

    def predict(self, n: int) -> wt.Predict:
        """Each case's labels from its first column (past its alignment) through its final window."""

        def labels(cases: Sequence[Any]) -> list[wt.Labels]:
            result: list[wt.Labels] = [() for _ in cases]
            for indices, rows in self._chunks(cases, n):
                predictions = self.network.classify_rows(self.network.prepare_dataset(rows))
                for i, prediction in zip(indices, predictions, strict=True):
                    result[i] = tuple(prediction[alignment(self.task, cases[i], n, self.k) :])
            return result

        return labels

    def evaluate(self, n: int, properties: Sequence[Any], limit: int | None = None) -> list[Any]:
        """The task's properties at n against the network (a screen of each when limit)."""
        return self.task.evaluate(self.predict(n), n, properties, limit)

    def codes_used(self, cases: Sequence[Any], n: int) -> set[int]:
        """
        The codes every state of these cases snaps to, the initial state's among them: a state of
        one factor's (factored states are the factored-state workplan's stage 7).
        """
        used: set[int] = set()
        for _, rows in self._chunks(cases, n):
            states = self.network.backend.matrix([tokens for tokens, _labels in rows])
            used.update(code for example in self.network.state_codes(states) for (code,) in example)
        return used


# the checks


def check_properties(task: wt.Task, n: int, catalogue: Sequence[Any]) -> list[Any]:
    """The catalogue at a check length: the held-out property (B3) only from HOLD_OUT_FROM (W24)."""
    return [p for p in catalogue if p.id != task.held_out_id or n >= HOLD_OUT_FROM]


def test_properties(task: wt.Task, catalogue: Sequence[Any]) -> list[Any]:
    """The catalogue at a test length: every test length is unseen whole, so no held-out property (W17)."""
    return [p for p in catalogue if p.id != task.held_out_id]


def break_length(passes: Mapping[int, bool]) -> int:
    """W10: the largest test length at which it and every shorter one pass; 0 when the first fails."""
    reached = 0
    for n in sorted(passes):
        if not passes[n]:
            break
        reached = n
    return reached


# runs


@dataclass(frozen=True)
class Settings:
    """A study's training, checking and test settings, the same for every run in it."""

    rate: float = 0.003
    epoch_examples: int = 2**16
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 30
    screen: int = 500
    steps: int = 3
    test_lengths: tuple[int, ...] = (8, 10, 12, 16, 24, 32, 48, 64)
    sampled: int = ad.SAMPLED  # the catalogue's sizes at a test length (W17)
    relation_sampled: int = ad.RELATION_SAMPLED
    per_stratum: int = ad.PER_STRATUM
    codes_cases: int = 1_000  # B1's first cases at each check and test length, for the codes used
    seeds: int = 5
    tests_past_break: bool = True  # False: a step's test pass ends at its first failing length (W12)

    def catalogue(self, task: wt.Task) -> tuple[Any, ...]:
        return task.catalogue(self.sampled, self.relation_sampled, self.per_stratum)


@dataclass(frozen=True)
class RunSpec:
    arm: str
    k: int
    s: int
    m: int | None  # None: a continuous state (E2)
    seed: int
    task: str = wt.ADDITION.name
    window: str = TRANSFORMER

    @property
    def key(self) -> str:
        """The run's name; an addition run's has no task, a transformer window's no window (the matrix's keys)."""
        prefix = "" if self.task == wt.ADDITION.name else f"{self.task}-"
        window = "" if self.window == TRANSFORMER else f"{self.window}-"
        state = "continuous" if self.m is None else f"m{self.m}"
        return f"{prefix}{self.arm}-{window}k{self.k}-s{self.s}-{state}-seed{self.seed}"

    @property
    def label(self) -> str:
        """The arm, with its window when it isn't the transformer (the report's cells)."""
        return self.arm if self.window == TRANSFORMER else f"{self.arm} ({self.window} window)"

    def step_counts(self, settings: Settings) -> list[list[int]]:
        """Each step's data window counts."""
        if self.arm == "all-at-once":
            return [list(range(1, settings.steps + 1))]
        return [list(range(1, j + 1)) for j in range(1, settings.steps + 1)]


def _provenance() -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {"commit": commit, "crate": crate_hash(), "machine": socket.gethostname()}


def _write_json(path: Path, value: Any) -> None:
    # atomic: a reader sees the old file or the whole new one
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1))
    os.replace(temporary, path)


def parameter_count(network: Any) -> int:
    return sum(math.prod(array.shape) for part in network.snapshot() for array in part)


def run_step(network: Any, spec: RunSpec, step: int, counts: Sequence[int], settings: Settings) -> dict[str, Any]:
    """One curriculum step: training to its stop, then its whole check and the test pass."""
    task = wt.task_named(spec.task)
    model = WindowModel(network, spec.k, task)
    catalogue = settings.catalogue(task)
    lengths = [w * spec.k for w in counts]
    checks = {n: check_properties(task, n, catalogue) for n in lengths}
    codes_property = next(p for p in catalogue if p.id == task.codes_id)
    screens: list[float] = []
    train_seconds: list[float] = []
    screen_seconds: list[float] = []
    best, since_best, stop, epochs = -1, 0, "budget", 0
    whole: dict[int, list[Any]] | None = None
    for epoch in range(settings.max_epochs):
        rng = random.Random(f"window-train-{spec.key}-{step}-{epoch}")
        start = time.perf_counter()
        train_epoch(
            network,
            training_examples(task, counts, spec.k, settings.epoch_examples, rng),
            settings.batch_size,
            settings.rate,
            rng,
        )
        train_seconds.append(time.perf_counter() - start)
        epochs = epoch + 1
        start = time.perf_counter()
        screen = [r for n in lengths for r in model.evaluate(n, checks[n], settings.screen)]
        passed = sum(r.passed for r in screen)
        screens.append(passed / sum(r.cases for r in screen))
        # a passing screen triggers the whole check: a spread of cases can miss a rare failure
        whole = {n: model.evaluate(n, checks[n]) for n in lengths} if all(r.holds for r in screen) else None
        screen_seconds.append(time.perf_counter() - start)
        if whole is not None and all(r.holds for results in whole.values() for r in results):
            stop = "passed"
            break
        if passed > best:
            best, since_best = passed, 0
        else:
            since_best += 1
            if since_best >= settings.patience:
                stop = "plateau"
                break
    start = time.perf_counter()
    trained = whole if stop == "passed" and whole is not None else {n: model.evaluate(n, checks[n]) for n in lengths}
    check_seconds = time.perf_counter() - start
    tests: dict[str, Any] = {}
    passes: dict[int, bool] = {}
    used: set[int] = set()
    snapped = spec.m is not None  # a continuous state has no codes to count

    def codes(n: int) -> set[int]:
        if not snapped:
            return set()
        return model.codes_used(task.property_cases(codes_property, n)[: settings.codes_cases], n)

    for n in lengths:
        used |= codes(n)
    for n in settings.test_lengths:
        start = time.perf_counter()
        results = model.evaluate(n, test_properties(task, catalogue))
        by_id = {r.id: r for r in results}
        test_codes = codes(n)
        used |= test_codes
        passes[n] = all(r.holds for r in results)
        tests[str(n)] = {
            "passes": passes[n],
            "reach": None if task.reach_id is None else task.reach(by_id[task.reach_id]),
            "codes": sorted(test_codes) if snapped else None,
            "properties": {r.id: task.result_json(r) for r in results},
            "seconds": time.perf_counter() - start,
        }
        if not passes[n] and not settings.tests_past_break:
            break
    reached = break_length(passes)
    return {
        "step": step,
        "counts": list(counts),
        "lengths": lengths,
        "epochs": epochs,
        "examples": epochs * settings.epoch_examples,
        "stop": stop,
        "passes": all(r.holds for results in trained.values() for r in results),
        "screens": screens,
        "checks": {str(n): {r.id: task.result_json(r) for r in results} for n, results in trained.items()},
        "tests": tests,
        "break": reached,
        "window_ratio": data_windows(reached, 0, spec.k) / max(counts),
        "codes_used": sorted(used) if snapped else None,
        "train_seconds": train_seconds,
        "screen_seconds": screen_seconds,
        "check_seconds": check_seconds,
    }


def train_run(spec: RunSpec, settings: Settings, directory: Path) -> dict[str, Any]:
    """
    One run's steps, each saved as it finishes under directory/<key>/ (step-<j>.json and its network,
    step-<j>.network.json); a run stopped part way resumes from its last saved step.
    """
    steps_directory = directory / spec.key
    steps_directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    network: Any = None
    steps: list[dict[str, Any]] = []
    for step, counts in enumerate(spec.step_counts(settings), start=1):
        saved = steps_directory / f"step-{step}.json"
        network_file = steps_directory / f"step-{step}.network.json"
        if saved.exists() and network_file.exists():
            steps.append(json.loads(saved.read_text()))
            network = None  # loaded when a later step needs it
            continue
        if network is None:
            network = (
                build(wt.task_named(spec.task), spec.k, spec.s, spec.m, spec.seed, spec.window)
                if step == 1
                else load_network(str(steps_directory / f"step-{step - 1}.network.json"))
            )
        result = run_step(network, spec, step, counts, settings)
        network.save(str(network_file))
        _write_json(saved, result)
        steps.append(result)
    last = steps[-1]
    return {
        "key": spec.key,
        "task": spec.task,
        "arm": spec.arm,
        "window": spec.window,
        "k": spec.k,
        "s": spec.s,
        "m": spec.m,
        "seed": spec.seed,
        "parameters": parameter_count(
            network or build(wt.task_named(spec.task), spec.k, spec.s, spec.m, spec.seed, spec.window)
        ),
        "steps": steps,
        "break": last["break"],
        "works": last["break"] >= WORKS,
        "seconds": time.perf_counter() - start,
        **_provenance(),
    }


# the study


@dataclass
class Study:
    directory: Path
    settings: Settings
    runs: list[RunSpec]
    workers: int
    train: Callable[[RunSpec, Settings, Path], dict[str, Any]] = field(default=train_run)

    @property
    def results(self) -> Path:
        return self.directory / "results"

    @property
    def models(self) -> Path:
        return self.directory / "models"


def load_config(path: Path) -> Study:
    """
    A study's JSON config: "directory" (results/ and models/ under it), "workers", "task" (default
    addition; window_tasks.task_named), the Settings' fields, and "matrix", a list of {"arm" (default
    "curriculum"), "window" (default "transformer"; WINDOWS), "k", "s", "m"}, each of the last three a
    list of values, every combination run with every seed; an m of null is a continuous state.
    """
    config = json.loads(path.read_text())
    fields = {k: config[k] for k in Settings.__dataclass_fields__ if k in config}
    if "test_lengths" in fields:
        fields["test_lengths"] = tuple(fields["test_lengths"])
    settings = Settings(**fields)
    task = wt.task_named(config.get("task", wt.ADDITION.name)).name
    runs: list[RunSpec] = []
    for cell in config["matrix"]:
        arm = cell.get("arm", "curriculum")
        assert arm in ARMS, f"an arm is one of {ARMS}; got {arm!r}"
        window = cell.get("window", TRANSFORMER)
        assert window in WINDOWS, f"a window is one of {tuple(WINDOWS)}; got {window!r}"
        runs += [
            RunSpec(arm, k, s, m, seed, task, window)
            for k in cell["k"]
            for s in cell["s"]
            for m in cell["m"]
            for seed in range(settings.seeds)
        ]
    return Study(Path(config["directory"]), settings, runs, config.get("workers", 1))


def read_results(results: Path) -> dict[str, dict[str, Any]]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(results.glob("*.json"))}


# C0: a rerun reproduces its run

PROVENANCE = ("commit", "crate", "machine", "attempts")


def comparable(result: dict[str, Any]) -> dict[str, Any]:
    """
    A result without what a rerun may change: its timings (every field named "seconds" or ending in
    "_seconds"), its provenance and attempts; a result from before the task field (E1's) is addition's,
    and one from before the window field a transformer window's.
    """

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            fields = cast("dict[str, Any]", value)
            return {k: strip(v) for k, v in fields.items() if k != "seconds" and not k.endswith("_seconds")}
        if isinstance(value, list):
            return [strip(v) for v in cast("list[Any]", value)]
        return value

    kept = strip({k: v for k, v in result.items() if k not in PROVENANCE})
    kept.setdefault("task", wt.ADDITION.name)
    kept.setdefault("window", TRANSFORMER)
    return kept


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    """Where two JSON values differ, as paths ("steps[2].break"); floats compared exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        first, second = cast("dict[str, Any]", a), cast("dict[str, Any]", b)
        return [
            found
            for key in sorted(first.keys() | second.keys())
            for found in (
                differences(first[key], second[key], f"{path}.{key}")
                if key in first and key in second
                else [f"{path}.{key}"]
            )
        ]
    if isinstance(a, list) and isinstance(b, list):
        first_items, second_items = cast("list[Any]", a), cast("list[Any]", b)
        if len(first_items) != len(second_items):
            return [f"{path} (length {len(first_items)} against {len(second_items)})"]
        return [
            found
            for i, (x, y) in enumerate(zip(first_items, second_items, strict=True))
            for found in differences(x, y, f"{path}[{i}]")
        ]
    same_type = type(cast("object", a)) is type(cast("object", b))
    return [] if same_type and a == b else [path or "."]


def compare_runs(original: Path, rerun: Path) -> dict[str, list[str]]:
    """
    C0: every run in both study directories, its result (comparable) and each step's saved network
    (weights, optimizer state and generator, exactly), against the original's; per run, the
    differences (none: reproduced).
    """
    first, second = read_results(original / "results"), read_results(rerun / "results")
    compared: dict[str, list[str]] = {}
    for key in sorted(first.keys() & second.keys()):
        found = differences(comparable(first[key]), comparable(second[key]), "result")
        networks = sorted((original / "models" / key).glob("step-*.network.json"))
        for network in networks:
            other = rerun / "models" / key / network.name
            if not other.exists():
                found.append(f"{network.name} (missing)")
                continue
            found += differences(
                json.loads(network.read_text()), json.loads(other.read_text()), network.name.removesuffix(".json")
            )
        compared[key] = found
    return compared


# E5: the codes a run's states reach past its training lengths (C6, unexplored states)

EXPLORED_LENGTHS = (8, 16, 64)


def explore_run(study: Study, spec: RunSpec, lengths: Sequence[int] = EXPLORED_LENGTHS) -> dict[str, Any] | None:
    """
    E5 for one finished, snapped run: the codes its last step's network snaps to over the first
    settings.codes_cases cases of the task's codes property (B1 for addition) at each of that step's
    check lengths and at each of lengths, and at each of lengths the codes never used at a check
    length ("new"); with whether the run works and whether its last check passed. None for a run
    without a result, one that raised, or a continuous one.
    """
    result_file = study.results / f"{spec.key}.json"
    if spec.m is None or not result_file.exists():
        return None
    result = json.loads(result_file.read_text())
    if "error" in result:
        return None
    last = result["steps"][-1]
    task = wt.task_named(spec.task)
    network = load_network(str(study.models / spec.key / f"step-{last['step']}.network.json"))
    model = WindowModel(network, spec.k, task)
    codes_property = next(p for p in study.settings.catalogue(task) if p.id == task.codes_id)

    def codes(n: int) -> list[int]:
        return sorted(model.codes_used(task.property_cases(codes_property, n)[: study.settings.codes_cases], n))

    checked = {n: codes(n) for n in last["lengths"]}
    seen = {code for used in checked.values() for code in used}
    reached = {n: codes(n) for n in lengths}
    return {
        "key": spec.key,
        "works": result["works"],
        "passes": last["passes"],
        "check_codes": {str(n): used for n, used in checked.items()},
        "codes": {str(n): used for n, used in reached.items()},
        "new": {str(n): [code for code in used if code not in seen] for n, used in reached.items()},
    }


def explore_summary(explored: Sequence[dict[str, Any]], lengths: Sequence[int] = EXPLORED_LENGTHS) -> str:
    """
    E5's comparison: of the runs whose last check passed, the working and the failing runs with a
    new code at each length, and Fisher's exact test between them (D4).
    """
    rows: list[list[str]] = []
    passing = [run for run in explored if run["passes"]]
    for n in lengths:
        counts = {
            works: (sum(bool(run["new"][str(n)]) for run in group), len(group))
            for works in (True, False)
            for group in [[run for run in passing if run["works"] is works]]
        }
        (working_new, working), (failing_new, failing) = counts[True], counts[False]
        p = inference.fisher_exact(failing_new, failing - failing_new, working_new, working - working_new)
        rows.append([str(n), f"{failing_new}/{failing}", f"{working_new}/{working}", f"{p:.3g}"])
    return table(["n", "failing runs with a new code", "working runs with a new code", "Fisher p"], rows)


# the findings' verdicts: works (W10), certified, robustly certified, the long tests


def network_name(path: Path) -> str:
    """A saved step's name, <run>-step-<j>: certify's and the long tests' file stem."""
    return f"{path.parent.name}-{path.name.removesuffix('.network.json')}"


def last_networks(study: Study) -> dict[str, Path]:
    """Each finished run's last step's network (the one its result's works describes), by run key."""
    results = read_results(study.results) if study.results.exists() else {}
    networks: dict[str, Path] = {}
    for spec in study.runs:
        result = results.get(spec.key)
        if result is not None and "error" not in result:
            networks[spec.key] = study.models / spec.key / f"step-{result['steps'][-1]['step']}.network.json"
    return networks


ROBUST, CERTIFIED, NOT_CERTIFIED = "robustly certified", "certified only strictly", "not certified"


def certificate_class(branched: Mapping[str, Any], strict: Mapping[str, Any]) -> str:
    """A network's class by its two certificates (D3): branched certified, strict only, or neither."""
    return ROBUST if branched["certified"] else CERTIFIED if strict["certified"] else NOT_CERTIFIED


def rate(successes: int, trials: int) -> str:
    """A rate with its Wilson 95% interval (D4); "-" without trials."""
    if trials == 0:
        return "-"
    low, high = inference.wilson(successes, trials)
    return f"{successes}/{trials} ({low:.2f}-{high:.2f})"


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def long_test_failures(long_test: Mapping[str, Any]) -> tuple[int, list[str]]:
    """A long test's cases in all, and each failing property and length with its cases wrong ("B1@1024 3")."""
    cases = 0
    failures: list[str] = []
    for n, length in long_test["lengths"].items():
        for property_id, result in length["properties"].items():
            cases += result["cases"]
            if result["passed"] != result["cases"]:
                failures.append(f"{property_id}@{n} {result['cases'] - result['passed']}")
    return cases, failures


LONG_LENGTHS = (128, 256, 1024)  # D9


def long_test(
    path: Path,
    task: wt.Task,
    lengths: Sequence[int] = LONG_LENGTHS,
    properties: Sequence[str] | None = None,
    sampled: int = ad.SAMPLED,
    per_stratum: int = ad.PER_STRATUM,
    doubling: bool = True,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    A saved network's tests at each length: its task's catalogue (sized; the stratified distances
    1, 2, 4, ..., n only when doubling, D9) without the held-out property, these properties only
    when given; each length's results and seconds, a line a property to log.
    """
    catalogue = task.catalogue(sampled, ad.RELATION_SAMPLED, per_stratum, doubling=doubling)
    tested = [p for p in test_properties(task, catalogue) if properties is None or p.id in properties]
    network = load_network(str(path))
    model = WindowModel(network, network.columns, task)
    results: dict[str, Any] = {}
    for n in lengths:
        start = time.perf_counter()
        checked = model.evaluate(n, tested)
        if log is not None:
            for result in checked:
                failure = "" if result.holds else f", e.g. {result.failures[0]}"
                log(f"n {n}, {result.id}: {result.passed}/{result.cases}{failure}")
        results[str(n)] = {
            "properties": {r.id: task.result_json(r) for r in checked},
            "seconds": time.perf_counter() - start,
        }
    return {"network": str(path), "lengths": results}


def long_test_file(
    path: Path, out: Path, task_name: str, lengths: Sequence[int], properties: Sequence[str] | None
) -> str:
    """
    The long tests of one network into out/<run>-step-<j>.json, unless that file is there already
    (a stopped pass resumes); a line of its cases and failures.
    """
    name = network_name(path)
    target = out / f"{name}.json"
    if target.exists():
        return f"{name}: done before"
    long = long_test(path, wt.task_named(task_name), lengths, properties)
    _write_json(target, long)
    cases, failures = long_test_failures(long)
    seconds = sum(length["seconds"] for length in long["lengths"].values())
    return f"{name}: {cases:,} cases, " + (", ".join(failures) or "every case passed") + f"; {seconds:.0f} s"


def verdicts(study: Study, branched: Path, strict: Path, long_tests: Path | None = None) -> str:
    """
    A study's findings tables from its results, its certificates (branched in one directory, strict
    in another, as certify writes them: <run>-step-<j>.json) and, when given, its long tests' JSON
    (check --json, the same names): per cell, the runs working (W10), certified and robustly
    certified, each with its Wilson interval, over the runs with a certificate; the runs by works
    and class; and per class the networks long-tested, with every failure.
    """
    results = read_results(study.results) if study.results.exists() else {}
    networks = last_networks(study)
    cells: dict[tuple[str, int, int, int | None], list[tuple[bool, str | None]]] = {}
    classes: dict[str, str] = {}
    for spec in study.runs:
        if spec.key not in networks:
            continue
        name = network_name(networks[spec.key])
        b, s = _read_json(branched / f"{name}.json"), _read_json(strict / f"{name}.json")
        kind = None if b is None or s is None else certificate_class(b, s)
        if kind is not None:
            classes[name] = kind
        cells.setdefault((spec.label, spec.k, spec.s, spec.m), []).append((results[spec.key]["works"], kind))

    def counts(done: Sequence[tuple[bool, str | None]]) -> list[str]:
        certified = [kind for _, kind in done if kind is not None]
        return [
            str(len(done)),
            rate(sum(works for works, _ in done), len(done)),
            rate(sum(kind != NOT_CERTIFIED for kind in certified), len(certified)),
            rate(sum(kind == ROBUST for kind in certified), len(certified)),
        ]

    ordered = sorted(cells.items(), key=lambda cell: (*cell[0][:3], cell[0][3] or 0))
    every = [run for _, done in ordered for run in done]
    rows = [
        [arm, str(k), str(s), "continuous" if m is None else str(m), *counts(done)] for (arm, k, s, m), done in ordered
    ]
    rows.append(["all", "", "", "", *counts(every)])
    sections = [table(["arm", "k", "s", "m", "runs", "working", "certified", "robustly certified"], rows)]
    by_works = [
        [kind, *(str(sum(1 for w, c in every if w is works and c == kind)) for works in (True, False))]
        for kind in (ROBUST, CERTIFIED, NOT_CERTIFIED)
    ]
    sections.append(table(["class", "working", "failing"], by_works))
    if long_tests is not None:
        tested: dict[str, list[tuple[str, int, list[str]]]] = {}
        for name, kind in classes.items():
            long_test = _read_json(long_tests / f"{name}.json")
            if long_test is not None:
                tested.setdefault(kind, []).append((name, *long_test_failures(long_test)))
        long_rows = [
            [
                kind,
                str(len(done)),
                str(sum(bool(failures) for _, _, failures in done)),
                f"{sum(cases for _, cases, _ in done):,}",
                "; ".join(f"{name}: {', '.join(failures)}" for name, _, failures in done if failures) or "-",
            ]
            for kind in (ROBUST, CERTIFIED, NOT_CERTIFIED)
            for done in [tested.get(kind, [])]
        ]
        sections.append(table(["class", "long-tested", "with a failure", "cases", "failures"], long_rows))
    return "\n\n".join(sections)


def _run_job(
    spec: RunSpec,
    directory: Path,
    settings: Settings,
    train: Callable[[RunSpec, Settings, Path], dict[str, Any]] = train_run,
) -> dict[str, Any]:
    """One run, its result written; a raise recorded as an error result."""
    results = directory / "results"
    marker = results / f"{spec.key}.running"
    marker.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    previous = results / f"{spec.key}.json"
    attempts = json.loads(previous.read_text()).get("attempts", 0) if previous.exists() else 0
    result: dict[str, Any]
    try:
        result = train(spec, settings, directory / "models")
    except Exception:  # noqa: BLE001 - any run that raises is recorded, and run once more
        result = {"key": spec.key, "error": traceback.format_exc()}
    result["attempts"] = attempts + 1
    _write_json(previous, result)
    marker.unlink(missing_ok=True)
    return result


def run(study: Study, log: Callable[[str], None] = print) -> dict[str, dict[str, Any]]:
    """
    Every run without a finished result, over study.workers processes (0: in this process); a run
    that raised is run once more.
    """
    study.results.mkdir(parents=True, exist_ok=True)
    study.models.mkdir(parents=True, exist_ok=True)
    for stale in study.results.glob("*.running"):
        stale.unlink()
    results = read_results(study.results)

    def todo(spec: RunSpec) -> bool:
        result = results.get(spec.key)
        return result is None or ("error" in result and result.get("attempts", 1) <= RETRIES)

    def record(result: dict[str, Any]) -> None:
        results[result["key"]] = result
        if "error" in result:
            log(f"{result['key']}: error (attempt {result['attempts']})")
            return
        steps = ", ".join(
            f"step {s['step']} {s['stop']} after {s['epochs']} epochs, break {s['break']}" for s in result["steps"]
        )
        log(f"{result['key']}: {'works' if result['works'] else 'fails'}; {steps}")

    while pending := [spec for spec in study.runs if todo(spec)]:
        if study.workers == 0:
            for spec in pending:
                record(_run_job(spec, study.directory, study.settings, study.train))
            continue
        assert study.train is train_run, "worker processes run train_run; workers=0 for another"
        context = multiprocessing.get_context("forkserver")
        with concurrent.futures.ProcessPoolExecutor(study.workers, mp_context=context) as pool:
            futures = {pool.submit(_run_job, spec, study.directory, study.settings): spec for spec in pending}
            for future in concurrent.futures.as_completed(futures):
                spec = futures[future]
                try:
                    record(future.result())
                except Exception:  # noqa: BLE001 - a worker that died: recorded, so the retry rule applies
                    previous = results.get(spec.key, {})
                    error = {
                        "key": spec.key,
                        "error": traceback.format_exc(),
                        "attempts": previous.get("attempts", 0) + 1,
                    }
                    _write_json(study.results / f"{spec.key}.json", error)
                    record(error)
    return results


# status and report


def status(study: Study) -> str:
    results = read_results(study.results) if study.results.exists() else {}
    errors = [k for k, r in results.items() if "error" in r]
    running = sorted(p.stem for p in study.results.glob("*.running")) if study.results.exists() else []
    steps_done = {spec.key: len(list((study.models / spec.key).glob("step-*.network.json"))) for spec in study.runs}
    steps = {spec.key: len(spec.step_counts(study.settings)) for spec in study.runs}
    works = sum(1 for r in results.values() if r.get("works"))
    lines = [
        (
            f"{len(results) - len(errors)} of {len(study.runs)} runs finished ({works} working), "
            f"{len(errors)} errors, {len(running)} running"
        ),
        *(f"running: {key}, {steps_done.get(key, 0)} of {steps.get(key, 0)} steps saved" for key in running),
        *(f"error: {key}" for key in errors),
    ]
    return "\n".join(lines)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _test_cell(step: Mapping[str, Any], n: int, field: str, form: str) -> str:
    """A test length's field in a report cell; "-" for a length past the break left untested."""
    test = step["tests"].get(str(n))
    return "-" if test is None else form.format(test[field])


def report(study: Study) -> str:
    """
    Per cell (arm, k, s, m): the seeds whose last step works, and per step the seeds passing its
    check, the break lengths and window ratios; per run and step, its stop, epochs, examples,
    break, ratio, L* at each test length and codes used; and the cost: seconds per epoch (training,
    screening) and each test length's pass; "-" for a length left untested (tests_past_break).
    """
    results = read_results(study.results) if study.results.exists() else {}
    seeds = study.settings.seeds
    cells: dict[tuple[str, int, int, int | None], list[dict[str, Any]]] = {}
    for spec in study.runs:
        result = results.get(spec.key)
        if result is not None and "error" not in result:
            cells.setdefault((spec.label, spec.k, spec.s, spec.m), []).append(result)
    summary: list[list[str]] = []
    runs: list[list[str]] = []
    cost: list[list[str]] = []
    lengths = study.settings.test_lengths
    for (arm, k, s, m), done in sorted(cells.items(), key=lambda cell: (*cell[0][:3], cell[0][3] or 0)):
        per_step = [
            f"{sum(r['steps'][j]['passes'] for r in done)} pass, breaks "
            + "/".join(str(r["steps"][j]["break"]) for r in done)
            for j in range(len(done[0]["steps"]))
        ]
        summary.append(
            [
                arm,
                str(k),
                str(s),
                "continuous" if m is None else str(m),
                str(done[0]["parameters"]),
                f"{sum(r['works'] for r in done)}/{len(done)}" + ("" if len(done) == seeds else f" of {seeds}"),
                "; ".join(f"step {j + 1}: {text}" for j, text in enumerate(per_step)),
            ]
        )
        for r in done:
            for step in r["steps"]:
                reach = " ".join(_test_cell(step, n, "reach", "{}") for n in lengths)
                runs.append(
                    [
                        r["key"],
                        str(step["step"]),
                        step["stop"],
                        str(step["epochs"]),
                        f"{step['examples']:,}",
                        "yes" if step["passes"] else "no",
                        str(step["break"]),
                        f"{step['window_ratio']:.2f}",
                        reach,
                        "-" if step["codes_used"] is None else str(len(step["codes_used"])),
                    ]
                )
                cost.append(
                    [
                        r["key"],
                        str(step["step"]),
                        f"{_median(step['train_seconds']):.1f}",
                        f"{_median(step['screen_seconds']):.1f}",
                        f"{step['check_seconds']:.1f}",
                        " ".join(_test_cell(step, n, "seconds", "{:.0f}") for n in lengths),
                    ]
                )
    test_header = "L* at " + " ".join(str(n) for n in lengths)
    return "\n\n".join(
        [
            table(["arm", "k", "s", "m", "parameters", "seeds working", "per step"], summary),
            table(
                ["run", "step", "stop", "epochs", "examples", "check passes", "break", "ratio", test_header, "codes"],
                runs,
            ),
            table(
                [
                    "run",
                    "step",
                    "s/epoch train",
                    "s/epoch screen",
                    "check s",
                    "test s at " + " ".join(str(n) for n in lengths),
                ],
                cost,
            ),
        ]
    )
