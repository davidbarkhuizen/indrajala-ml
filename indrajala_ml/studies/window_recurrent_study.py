"""
The window-recurrent adder's harness (docs/window-recurrent-adder-workplan.md, stage 8;
scripts/window_recurrent_study.py).

A run is one arm, window size k, state width s, code count m and seed: W11's window (d 16, 3
causal attention blocks of 2 heads, each read the state through a StateInput, FFN width 32) beside
a WindowRecurrence(s, Codes(m)), on the Rust backend, under Adam.

An addition of n columns is read at an alignment a (W6-i): a zero columns, the n column tokens
(candidate 1's, 9 a_t + 3 b_t + c_t), and zero columns up to whole windows, then one window more
(W9): ceil((n + a) / k) + 1 windows, a's data windows ceil((n + a) / k). Its labels are the sum's
digits from column a, the zero columns below labelled 0. In training a is uniform in 0..k - 1;
in a check it is the case's own, from a hash of the triple (W23), the same for every model.

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indrajala_ml.data import addition_data as ad
from indrajala_ml.data.addition_data import BASE, Triple
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
from indrajala_ml.model.specs.window_recurrence import Codes, WindowRecurrence
from indrajala_ml.studies.addition_study import column_tokens
from indrajala_ml.studies.common import crate_hash, table

D, BLOCKS, HEADS, FFN = 16, 3, 2, 32  # W11
HOLD_OUT_FROM = 3  # W24: below it every triple is trained on, and B3 isn't checked
RETRIES = 1
WORKS = 16  # W10: a model works when its break length is at least this
ARMS = ("curriculum", "all-at-once")

Example = tuple[tuple[float, ...], tuple[int, ...]]


def window_specs() -> list[LayerSpec]:
    """W11's window: the state entering before each attention block (W18)."""
    blocks: list[LayerSpec] = []
    for _ in range(BLOCKS):
        blocks += [
            StateInput(),
            Residual((LayerNorm(), Attention(heads=HEADS, causal=True))),
            Residual((LayerNorm(), Dense(FFN, activation="relu"), Dense(D, activation="linear", bias=True))),
        ]
    return [
        Embedding(BASE**3, D),
        Position(),
        *blocks,
        LayerNorm(),
        Dense(BASE, output=True, activation="softmax", loss="cross_entropy"),
    ]


def build(k: int, s: int, m: int, seed: int) -> Any:
    return WindowRecurrentRustArrayNetwork.randomized(
        (k,), window_specs(), WindowRecurrence(s, Codes(m)), Adam(), seed=seed
    )


# the encoding


def data_windows(n: int, a: int, k: int) -> int:
    return -(-(n + a) // k)


def alignment(triple: Triple, n: int, k: int) -> int:
    """W23: a case's alignment, from the high half of its hash (the low half decides D8's tenth)."""
    return (ad.triple_hash(triple, n) >> 32) % k


def example(triple: Triple, n: int, k: int, a: int) -> Example:
    """The triple's columns at alignment a, through its final zero window, and their sum digits."""
    width = (data_windows(n, a, k) + 1) * k
    columns = (0,) * a + column_tokens(triple, n)[:n]
    tokens = columns + (0,) * (width - len(columns))
    return tuple(float(t) for t in tokens), (0,) * a + ad.digits(sum(triple), width - a)


def lengths_for(windows: int, a: int, k: int) -> range:
    """The lengths n >= 1 that fill exactly this many data windows at alignment a."""
    return range(max(1, (windows - 1) * k - a + 1), windows * k - a + 1)


def training_examples(counts: Sequence[int], k: int, count: int, rng: random.Random) -> list[Example]:
    """count examples, an equal share per data window count, D7's three kinds in turn within each."""
    examples: list[Example] = []
    for i in range(count):
        windows = counts[i % len(counts)]
        a = rng.randrange(k)
        n = rng.choice(lengths_for(windows, a, k))
        triple = ad.training_triple(i // len(counts), n, rng, hold_out=n >= HOLD_OUT_FROM)
        examples.append(example(triple, n, k, a))
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


class WindowAdder:
    """A window-recurrent network as the catalogue's Adder at a length n."""

    def __init__(self, network: Any, k: int) -> None:
        self.network, self.k = network, k

    def _examples(self, triples: Sequence[Triple], n: int) -> list[Example]:
        return [example(t, n, self.k, alignment(t, n, self.k)) for t in triples]

    def adder(self, n: int) -> ad.Adder:
        def add(triples: Sequence[Triple]) -> list[int]:
            examples = self._examples(triples, n)
            sums = [0] * len(triples)
            for indices in _by_width(examples).values():
                rows = [examples[i] for i in indices]
                predictions = self.network.classify_rows(self.network.prepare_dataset(rows))
                for i, prediction in zip(indices, predictions, strict=True):
                    sums[i] = ad.from_digits(prediction[alignment(triples[i], n, self.k) :])
            return sums

        return add

    def codes_used(self, triples: Sequence[Triple], n: int) -> set[int]:
        """The codes every state of these additions snaps to, the initial state's among them."""
        examples = self._examples(triples, n)
        used: set[int] = set()
        for indices in _by_width(examples).values():
            states = self.network.backend.matrix([examples[i][0] for i in indices])
            used.update(code for codes in self.network.state_codes(states) for code in codes)
        return used


# the checks


def check_properties(n: int, catalogue: Sequence[ad.Property]) -> list[ad.Property]:
    """The catalogue at a check length: B3 only from HOLD_OUT_FROM (W24)."""
    return [p for p in catalogue if p.id != "B3" or n >= HOLD_OUT_FROM]


def test_properties(catalogue: Sequence[ad.Property]) -> list[ad.Property]:
    """The catalogue at a test length: every test length is unseen whole, so no B3 (W17)."""
    return [p for p in catalogue if p.id != "B3"]


def _result_json(result: ad.PropertyResult) -> dict[str, Any]:
    return {
        "cases": result.cases,
        "passed": result.passed,
        "strata": {str(k): list(v) for k, v in sorted(result.strata.items())},
        "failures": [{"triple": list(c.triple), "column": c.column} for c in result.failures],
    }


def _evaluate(
    model: WindowAdder, n: int, properties: Sequence[ad.Property], limit: int | None = None
) -> list[ad.PropertyResult]:
    return ad.evaluate(model.adder(n), n, properties, limit=limit, shrink=False)


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

    def catalogue(self) -> tuple[ad.Property, ...]:
        return ad.catalogue(self.sampled, self.relation_sampled, self.per_stratum)


@dataclass(frozen=True)
class RunSpec:
    arm: str
    k: int
    s: int
    m: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.arm}-k{self.k}-s{self.s}-m{self.m}-seed{self.seed}"

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
    model = WindowAdder(network, spec.k)
    catalogue = settings.catalogue()
    lengths = [w * spec.k for w in counts]
    checks = {n: check_properties(n, catalogue) for n in lengths}
    b1 = next(p for p in catalogue if p.id == "B1")
    screens: list[float] = []
    train_seconds: list[float] = []
    screen_seconds: list[float] = []
    best, since_best, stop, epochs = -1, 0, "budget", 0
    whole: dict[int, list[ad.PropertyResult]] | None = None
    for epoch in range(settings.max_epochs):
        rng = random.Random(f"window-train-{spec.key}-{step}-{epoch}")
        start = time.perf_counter()
        train_epoch(
            network,
            training_examples(counts, spec.k, settings.epoch_examples, rng),
            settings.batch_size,
            settings.rate,
            rng,
        )
        train_seconds.append(time.perf_counter() - start)
        epochs = epoch + 1
        start = time.perf_counter()
        screen = [r for n in lengths for r in _evaluate(model, n, checks[n], settings.screen)]
        passed = sum(r.passed for r in screen)
        screens.append(passed / sum(r.cases for r in screen))
        # a passing screen triggers the whole check: a spread of cases can miss a rare failure
        whole = {n: _evaluate(model, n, checks[n]) for n in lengths} if all(r.holds for r in screen) else None
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
    trained = whole if stop == "passed" and whole is not None else {n: _evaluate(model, n, checks[n]) for n in lengths}
    check_seconds = time.perf_counter() - start
    tests: dict[str, Any] = {}
    passes: dict[int, bool] = {}
    used: set[int] = set()

    def codes(n: int) -> set[int]:
        return model.codes_used([c.triple for c in ad.property_cases(b1, n)[: settings.codes_cases]], n)

    for n in lengths:
        used |= codes(n)
    for n in settings.test_lengths:
        start = time.perf_counter()
        results = _evaluate(model, n, test_properties(catalogue))
        by_id = {r.id: r for r in results}
        test_codes = codes(n)
        used |= test_codes
        passes[n] = all(r.holds for r in results)
        tests[str(n)] = {
            "passes": passes[n],
            "reach": by_id["M3"].reach(),
            "codes": sorted(test_codes),
            "properties": {r.id: _result_json(r) for r in results},
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
        "checks": {str(n): {r.id: _result_json(r) for r in results} for n, results in trained.items()},
        "tests": tests,
        "break": reached,
        "window_ratio": data_windows(reached, 0, spec.k) / max(counts),
        "codes_used": sorted(used),
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
                build(spec.k, spec.s, spec.m, spec.seed)
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
        "arm": spec.arm,
        "k": spec.k,
        "s": spec.s,
        "m": spec.m,
        "seed": spec.seed,
        "parameters": parameter_count(network or build(spec.k, spec.s, spec.m, spec.seed)),
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
    A study's JSON config: "directory" (results/ and models/ under it), "workers", the Settings'
    fields, and "matrix", a list of {"arm" (default "curriculum"), "k", "s", "m"}, each a list of
    values, every combination run with every seed.
    """
    config = json.loads(path.read_text())
    fields = {k: config[k] for k in Settings.__dataclass_fields__ if k in config}
    if "test_lengths" in fields:
        fields["test_lengths"] = tuple(fields["test_lengths"])
    settings = Settings(**fields)
    runs: list[RunSpec] = []
    for cell in config["matrix"]:
        arm = cell.get("arm", "curriculum")
        assert arm in ARMS, f"an arm is one of {ARMS}; got {arm!r}"
        runs += [
            RunSpec(arm, k, s, m, seed)
            for k in cell["k"]
            for s in cell["s"]
            for m in cell["m"]
            for seed in range(settings.seeds)
        ]
    return Study(Path(config["directory"]), settings, runs, config.get("workers", 1))


def read_results(results: Path) -> dict[str, dict[str, Any]]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(results.glob("*.json"))}


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
    cells: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for spec in study.runs:
        result = results.get(spec.key)
        if result is not None and "error" not in result:
            cells.setdefault((spec.arm, spec.k, spec.s, spec.m), []).append(result)
    summary: list[list[str]] = []
    runs: list[list[str]] = []
    cost: list[list[str]] = []
    lengths = study.settings.test_lengths
    for (arm, k, s, m), done in sorted(cells.items()):
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
                str(m),
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
                        str(len(step["codes_used"])),
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
