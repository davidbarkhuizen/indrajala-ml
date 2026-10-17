"""
Window-recurrent networks timed on the Rust backend: the training step, the forward pass and a
long test (the window-program workplan's stage 0). A case is a name, its kind first:

- `learn-<window>-k<k>-s<s>-m<m>-B<B>-W<W>`: `learn_batch` of the window-recurrent study's network
  (`window_recurrent_study.build`: the window, `transformer` or `linear`, beside a state of s
  features and m codes), a batch of B examples of W windows of k columns, random column tokens and
  digits (the time doesn't depend on them). The old name `k<k>-s<s>-m<m>-B<B>-W<W>` is the
  transformer's (so the cases timed before stage 0 keep their rows).
- `learn-family-p<p>-f<m>-B<B>-W<W>`: `learn_batch` of the division family's network
  (`modular_family.network`: the transformer window reading its output, two factors of `s` 16 and m
  codes; the sum and difference queried), B examples of W columns (`W - 1` uniform and the final zero).
- `forward-linear-k1-s<s>-m<m>-n<n>` and `forward-family-p<p>-f<m>-n<n>`: `classify_rows` on one
  long-test chunk at n columns (`CHUNK_TOKENS // ((n + 1) * width)` rows of `n + 1` windows, or r
  rows with a field `r<r>`), the dataset prepared once before timing.
- `long-test-family-p<p>-f<m>-U1-n<n>-c<c>`: the long test end to end at n columns, c cases of U1
  through `QueryModel.predict` and `task.evaluate` (the examples, the network, the labels, the
  checking), the network built from seed 0 (the time doesn't depend on its answers).
- `study-family-p<p>-f<m>-e<e>`: a short study run end to end, `modular_family.run` of the family
  arm (two factors of m codes, the window's output read) from seed 0 for e epochs in all, into a
  temporary directory (the window-program workplan's stage 6: a real run, not one step).

Each case is built from a fixed seed, warmed up, then timed in repeats of loops steps (a learn
case's loops `--loops`; a forward or long-test case's 1, each step seconds long); each repeat prints
a row, its seconds per step. A fixed crate matmul is timed the same way as the control, a path no
window-recurrent change touches.

    python scripts/window_recurrent_timing.py [--cases CASE ...] [--repeats 5] [--loops 20]
    python scripts/window_recurrent_timing.py --breakdown CASE [CASE ...]

It prints the probe contract's lines (scripts/ab.py, `--bench cmd`), one JSON object per row:

    {"case": "learn linear k1 s16 m16 B64 W19", "metric": "per step", "value": 0.0123, "unit": "s"}

`--breakdown` times no rows: it runs each case's step once unprofiled (its wall time) and once under
cProfile, and prints the step's components: the crate's own time (every `indrajala_math_rust`
function's), the network's call (`learn_batch` or `classify_rows`) less the crate's, so the window
loop's Python with each crate call's Python side, and, for a long test, the examples
(`Encoding.example`, `column_tokens`), `prepare_dataset` and the checking (`evaluate` less
`predict`'s chunks). cProfile charges Python calls more than crate calls, so the Python shares are
overstated: an upper bound for the window-program workplan's D10 gate.

ab.py runs this script from the new tree on both sides:

    python scripts/ab.py remote --host jebel run --bench cmd --old OLD --new NEW --order ONNO \\
        -- scripts/window_recurrent_timing.py
"""

import argparse
import cProfile
import json
import os
import pstats
import random
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any, cast

# run as `python scripts/window_recurrent_timing.py` from the repo root, which puts scripts/ (not the
# repo root) on sys.path: the root goes last, a fallback, so a PYTHONPATH tree comes first (ab.py
# runs the new tree's script with each side's tree on PYTHONPATH)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# the study's cells: k 1 (E7, E8, C9's) and k 4 (stage 7's), the state and codes of E8b; E2b''s
# linear window (its longest training length 18, and the final zero window) and the division
# family's fork probe (p 7, two factors of 32 codes, certified at steps 9-15: 15 columns and the
# zero); the long tests' forward pass and a long test at 1,024 (1,023 cases: three full chunks)
DEFAULT_CASES = [
    "k1-s4-m16-B32-W12",
    "k4-s4-m16-B32-W5",
    "learn-linear-k1-s16-m16-B64-W19",
    "learn-family-p7-f32-B64-W16",
    "forward-linear-k1-s16-m16-n1024",
    "forward-family-p7-f32-n1024",
    "long-test-family-p7-f32-U1-n1024-c1023",
]

KINDS = ("learn", "forward", "long-test", "study")
FAMILY_QUERIES = ("sum", "difference")
FAMILY_STATE = 16  # s of each factor, the division studies'

# pstats' table: a function (file, line, name) to its calls, primitive calls, own and cumulative time, callers
Function = tuple[str, int, str]
Profiled = dict[Function, tuple[int, int, float, float, dict[Function, Any]]]


def parse_case(case: str) -> tuple[str, str, dict[str, int], str | None]:
    """'learn-linear-k1-s16-m16-B64-W19' to (kind, window, its numeric fields, its property)."""
    kind = next((k for k in KINDS if case.startswith(k + "-")), "learn")
    parts = case.removeprefix(kind + "-").split("-") if case.startswith(kind + "-") else case.split("-")
    window = parts.pop(0) if parts[0] in ("transformer", "linear", "family") else "transformer"
    prop = next((p for p in parts if p[0] == "U"), None)
    fields = {p[0]: int(p[1:]) for p in parts if p != prop}
    return kind, window, fields, prop


def timed(step: Callable[[], object], repeats: int, loops: int) -> list[float]:
    """Seconds per step in each repeat, after a warm-up of loops steps."""
    for _ in range(loops):
        step()
    seconds: list[float] = []
    for _ in range(repeats):
        begun = time.perf_counter()
        for _ in range(loops):
            step()
        seconds.append((time.perf_counter() - begun) / loops)
    return seconds


def family(fields: dict[str, int]) -> tuple[Any, Any]:
    """The division family's encoding and network: p, two factors of s 16 and f codes, reading the output."""
    from indrajala_ml.model.specs.window_recurrence import OUTPUT
    from indrajala_ml.studies import modular_family as mf

    encoding = mf.Encoding(fields["p"], FAMILY_QUERIES)
    factors = [(FAMILY_STATE, fields["f"])] * 2
    return encoding, mf.network(encoding, factors, 0, reads=OUTPUT)


def addition_rows(network: Any, k: int, batch: int, windows: int) -> list[tuple[tuple[float, ...], tuple[int, ...]]]:
    """B examples of W windows of k columns: random column tokens and digits."""
    rng = np.random.default_rng(0)
    vocabulary = network.layer_specs[0].vocabulary
    tokens = rng.integers(0, vocabulary, size=(batch, windows * k))
    digits = rng.integers(0, network.class_count, size=(batch, windows * k))
    return [(tuple(float(t) for t in x), tuple(int(d) for d in y)) for x, y in zip(tokens, digits, strict=True)]


def family_rows(encoding: Any, columns: int, batch: int) -> list[Any]:
    """B examples of the given columns each (the last the final zero)."""
    from indrajala_ml.studies import modular_family as mf

    rng = random.Random(0)
    return [encoding.example(mf.columns(columns - 1, rng)) for _ in range(batch)]


def chunk_rows(n: int, width: int) -> int:
    """A long test's rows a forward pass at n columns (the harness's chunk)."""
    from indrajala_ml.studies import window_recurrent_study as wr

    return max(1, wr.CHUNK_TOKENS // ((n + 1) * width))


def learn_step(window: str, fields: dict[str, int]) -> Callable[[], object]:
    """One training step on a fixed batch."""
    from indrajala_ml.studies import window_recurrent_study as wr
    from indrajala_ml.studies import window_tasks as wt

    if window == "family":
        encoding, network = family(fields)
        rows = family_rows(encoding, fields["W"], fields["B"])
    else:
        k = fields["k"]
        network = wr.build(wt.ADDITION, k, fields["s"], fields["m"], seed=0, window=window)
        rows = addition_rows(network, k, fields["B"], fields["W"])
    return lambda: network.learn_batch(0.001, rows)


def forward_step(window: str, fields: dict[str, int]) -> Callable[[], object]:
    """One long-test chunk's forward pass, its dataset prepared."""
    from indrajala_ml.studies import window_recurrent_study as wr
    from indrajala_ml.studies import window_tasks as wt

    n = fields["n"]
    if window == "family":
        encoding, network = family(fields)
        rows = family_rows(encoding, n + 1, fields.get("r") or chunk_rows(n, encoding.width))
    else:
        k = fields["k"]
        network = wr.build(wt.ADDITION, k, fields["s"], fields["m"], seed=0, window=window)
        rows = addition_rows(network, k, fields.get("r") or chunk_rows(n, k), -(-(n + 1) // k))
    prepared = network.prepare_dataset(rows)
    return lambda: network.classify_rows(prepared)


def long_test_step(fields: dict[str, int], prop: str) -> Callable[[], object]:
    """A long test at n columns: c cases of the property, examples to checked results."""
    from indrajala_ml.data import addition_data as ad
    from indrajala_ml.studies import modular_family as mf
    from indrajala_ml.studies import window_recurrent_study as wr
    from indrajala_ml.studies import window_tasks as wt

    encoding, network = family(fields)
    task = wt.task_named(f"modular-division-{fields['p']}")
    catalogue = task.catalogue(fields["c"], ad.RELATION_SAMPLED, ad.PER_STRATUM, doubling=True)
    tested = [p for p in wr.test_properties(task, catalogue) if p.id == prop]
    model = mf.QueryModel(network, encoding, task, mf.slot(task.name))
    n = fields["n"]
    return lambda: task.evaluate(model.predict(n), n, tested)


def study_step(fields: dict[str, int]) -> Callable[[], object]:
    """A short family probe run of e epochs, each run into its own temporary directory."""
    import tempfile
    from pathlib import Path

    from indrajala_ml.model.specs.window_recurrence import OUTPUT
    from indrajala_ml.studies import modular_family as mf

    encoding = mf.Encoding(fields["p"], FAMILY_QUERIES)
    factors = [(16, fields["f"]), (16, fields["f"])]

    def step() -> None:
        with tempfile.TemporaryDirectory() as out:
            mf.run(encoding, factors, 0, 1e9, Path(out), reads=OUTPUT, budget_epochs=fields["e"])

    return step


def case_step(case: str) -> tuple[Callable[[], object], bool]:
    """A case's step, and whether it is an inference step (one loop a repeat)."""
    kind, window, fields, prop = parse_case(case)
    if kind == "learn":
        return learn_step(window, fields), False
    if kind == "forward":
        return forward_step(window, fields), True
    if kind == "study":
        return study_step(fields), True
    assert prop is not None, f"{case}: a long test names its property"
    return long_test_step(fields, prop), True


def control_step() -> Callable[[], object]:
    """A fixed crate matmul, (256, 256) by (256, 256): untouched by window-recurrent changes."""
    import indrajala_math_rust as pa

    rng = np.random.default_rng(0)
    a, b = (pa.Array(rng.uniform(-1, 1, (256, 256)).tolist()) for _ in range(2))
    return lambda: a @ b


def breakdown(case: str) -> list[str]:
    """A case's step once unprofiled and once under cProfile, its components (the module docstring)."""
    step, _ = case_step(case)
    step()  # warm-up
    begun = time.perf_counter()
    step()
    wall = time.perf_counter() - begun
    profile = cProfile.Profile()
    profile.runcall(step)
    stats = cast(Profiled, pstats.Stats(profile).stats)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    # the step's own cumulative time: the functions' own times sum to a little less (the profiler's overhead)
    total = max(ct for _, _, _, ct, _ in stats.values())

    def named(name: str) -> bool:
        return "indrajala_math_rust" in name

    def cumulative(*functions: str) -> float:
        """The named functions' cumulative time, the outermost calls only (a wrapper's same-named inner call not again)."""
        matching = {key for key in stats if key[2] in functions}
        return sum(stats[key][3] for key in matching if not matching & set(stats[key][4]))

    crate = sum(tt for (file, _, f), (_, _, tt, _, _) in stats.items() if named(file) or named(f))
    network = cumulative("learn_batch", "classify_rows")
    parts = {
        "crate (its own time)": crate,
        "network call less the crate (window loop, per-call Python)": network - crate,
    }
    if case.startswith("long-test"):
        examples = cumulative("example", "column_tokens")
        prepare = cumulative("prepare_dataset")
        parts["examples"] = examples
        parts["prepare_dataset"] = prepare
        parts["the rest (cases, labels' read-back, checking)"] = total - network - examples - prepare
    else:
        parts["the rest"] = total - network
    lines = [f"{case}: {wall:.3f} s unprofiled, {total:.3f} s profiled"]
    lines += [f"  {name}: {seconds:.3f} s, {seconds / total:.1%}" for name, seconds in parts.items()]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--loops", type=int, default=20, help="a learn case's steps a repeat")
    parser.add_argument("--breakdown", nargs="+", metavar="CASE", help="profile these cases; no rows")
    arguments = parser.parse_args()
    if arguments.breakdown:
        for case in arguments.breakdown:
            print("\n".join(breakdown(case)), flush=True)
        return
    steps = [(case.replace("-", " "), *case_step(case), False) for case in arguments.cases]
    steps.append(("matmul 256", control_step(), False, True))
    for name, step, inference, control in steps:
        seconds = timed(step, arguments.repeats, 1 if inference else arguments.loops)
        for value in seconds:
            row = {"case": name, "metric": "per step", "value": value, "unit": "s"}
            print(json.dumps({**row, "control": True} if control else row), flush=True)
        print(f"# {name}: median {statistics.median(seconds):.6f} s per step", file=sys.stderr)


if __name__ == "__main__":
    main()
