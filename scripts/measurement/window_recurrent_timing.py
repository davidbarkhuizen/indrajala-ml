"""
Window-recurrent networks timed on the Rust backend: the training step, the forward pass and a
long test (the window-program workplan's stage 0). A case is a name, its kind first:

- `learn-<window>-k<k>-s<s>-m<m>-B<B>-W<W>`: `learn_batch` of the window-recurrent adder's network
  (the window preset, `transformer` or `linear` (indrajala_ml/model/specs/window_presets.py), over
  the adder's 27 column tokens and 3 digits, beside a state of s features and m codes), a batch of
  B examples of W windows of k columns, random column tokens and digits (the time doesn't depend on
  them). The old name `k<k>-s<s>-m<m>-B<B>-W<W>` is the
  transformer's (so the cases timed before stage 0 keep their rows).
- `learn-family-p<p>-f<m>-B<B>-W<W>`: `learn_batch` of the division family's network (the
  transformer window over the family's 11 tokens and p + 1 classes, three tokens a column (a digit,
  the sum's and the difference's queries), two factors of `s` 16 and m codes reading the window's
  output), B examples of W columns, random tokens and labels (the time doesn't depend on them).
- `forward-linear-k1-s<s>-m<m>-n<n>` and `forward-family-p<p>-f<m>-n<n>`: `classify_rows` on one
  long-test chunk at n columns (`CHUNK_TOKENS // ((n + 1) * width)` rows of `n + 1` windows, or r
  rows with a field `r<r>`), the dataset prepared once before timing.

The end-to-end cases, a long test and a short study run (`long-test-family-...`, `study-family-...`),
run the studies' own code, so they are indrajala-studies' `scripts/measurement/window_study_timing.py`,
which reuses this script's parsing, timing and breakdown; ab.py runs it by its absolute path.

Each case is built from a fixed seed, warmed up, then timed in repeats of loops steps (a learn
case's loops `--loops`; a forward or long-test case's 1, each step seconds long); each repeat prints
a row, its seconds per step. A fixed crate matmul is timed the same way as the control, a path no
window-recurrent change touches.

    python scripts/measurement/window_recurrent_timing.py [--cases CASE ...] [--repeats 5] [--loops 20]
    python scripts/measurement/window_recurrent_timing.py --breakdown CASE [CASE ...]

It prints the probe contract's lines (scripts/measurement/ab.py, `--bench cmd`), one JSON object per row:

    {"case": "learn linear k1 s16 m16 B64 W19", "metric": "per step", "value": 0.0123, "unit": "s"}

`--breakdown` times no rows: it runs each case's step once unprofiled (its wall time) and once under
cProfile, and prints the step's components: the crate's own time (every `indrajala_math_rust`
function's), the network's call (`learn_batch` or `classify_rows`) less the crate's, so the window
loop's Python with each crate call's Python side, and, for a long test, the examples
(`Encoding.example`, `column_tokens`), `prepare_dataset` and the checking (`evaluate` less
`predict`'s chunks). cProfile charges Python calls more than crate calls, so the Python shares are
overstated: an upper bound for the window-program workplan's D10 gate.

ab.py runs this script from the new tree on both sides:

    python scripts/measurement/ab.py remote --host jebel run --bench cmd --old OLD --new NEW --order ONNO \\
        -- scripts/measurement/window_recurrent_timing.py
"""

import argparse
import cProfile
import json
import os
import pstats
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any, cast

# run as `python scripts/measurement/window_recurrent_timing.py` from the repo root, which puts scripts/ (not the
# repo root) on sys.path: the root goes last, a fallback, so a PYTHONPATH tree comes first (ab.py
# runs the new tree's script with each side's tree on PYTHONPATH)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from indrajala_ml.model.networks.window_recurrent_array_network import WindowRecurrentRustArrayNetwork
from indrajala_ml.model.specs import window_presets as wp
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import OUTPUT, AttentionPooling, Codes, Factor, WindowRecurrence

# the study's cells: k 1 (E7, E8, C9's) and k 4 (stage 7's), the state and codes of E8b; E2b''s
# linear window (its longest training length 18, and the final zero window) and the division
# family's fork probe (p 7, two factors of 32 codes, certified at steps 9-15: 15 columns and the
# zero); the long tests' forward pass at 1,024 (the long test itself is window_study_timing.py's)
DEFAULT_CASES = [
    "k1-s4-m16-B32-W12",
    "k4-s4-m16-B32-W5",
    "learn-linear-k1-s16-m16-B64-W19",
    "learn-family-p7-f32-B64-W16",
    "forward-linear-k1-s16-m16-n1024",
    "forward-family-p7-f32-n1024",
]

KINDS = ("learn", "forward", "long-test", "study")
FAMILY_STATE = 16  # s of each factor, the division studies'
# the window-recurrent adder's column tokens and digits (window_tasks.ADDITION's machine), and the
# division family's tokens and tokens a column, the sum and difference queried (modular_family.Encoding)
ADDITION_TOKENS, ADDITION_CLASSES = 27, 3
FAMILY_TOKENS, FAMILY_WIDTH = 11, 3
CHUNK_TOKENS = 1 << 20  # a long test's forward pass at most (window_recurrent_study.CHUNK_TOKENS)
WINDOWS = {"transformer": wp.transformer_window, "linear": wp.linear_window}

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


def adder(window: str, fields: dict[str, int]) -> Any:
    """The window-recurrent adder's network: the window over k columns, a state of s features and m codes, seed 0."""
    recurrence = WindowRecurrence((Factor(fields["s"], Codes(fields["m"])),))
    specs = WINDOWS[window](ADDITION_TOKENS, ADDITION_CLASSES)
    return WindowRecurrentRustArrayNetwork.randomized((fields["k"],), specs, recurrence, Adam(), seed=0)


def family(fields: dict[str, int]) -> Any:
    """The division family's network: p, two factors of s 16 and f codes, reading the window's output, seed 0."""
    factors = (Factor(FAMILY_STATE, Codes(fields["f"])),) * 2
    recurrence = WindowRecurrence(factors, False, pooling=AttentionPooling(OUTPUT))
    specs = wp.transformer_window(FAMILY_TOKENS, fields["p"] + 1)
    return WindowRecurrentRustArrayNetwork.randomized((FAMILY_WIDTH,), specs, recurrence, Adam(), seed=0)


def random_rows(network: Any, batch: int, tokens: int) -> list[tuple[tuple[float, ...], tuple[int, ...]]]:
    """B examples of the given tokens each: random tokens and labels."""
    rng = np.random.default_rng(0)
    vocabulary = network.layer_specs[0].vocabulary
    xs = rng.integers(0, vocabulary, size=(batch, tokens))
    labels = rng.integers(0, network.class_count, size=(batch, tokens))
    return [(tuple(float(t) for t in x), tuple(int(d) for d in y)) for x, y in zip(xs, labels, strict=True)]


def chunk_rows(n: int, width: int) -> int:
    """A long test's rows a forward pass at n columns (the harness's chunk)."""
    return max(1, CHUNK_TOKENS // ((n + 1) * width))


def learn_step(window: str, fields: dict[str, int]) -> Callable[[], object]:
    """One training step on a fixed batch."""
    if window == "family":
        network = family(fields)
        rows = random_rows(network, fields["B"], fields["W"] * FAMILY_WIDTH)
    else:
        network = adder(window, fields)
        rows = random_rows(network, fields["B"], fields["W"] * fields["k"])
    return lambda: network.learn_batch(0.001, rows)


def forward_step(window: str, fields: dict[str, int]) -> Callable[[], object]:
    """One long-test chunk's forward pass, its dataset prepared."""
    n = fields["n"]
    if window == "family":
        network = family(fields)
        rows = random_rows(network, fields.get("r") or chunk_rows(n, FAMILY_WIDTH), (n + 1) * FAMILY_WIDTH)
    else:
        network, k = adder(window, fields), fields["k"]
        rows = random_rows(network, fields.get("r") or chunk_rows(n, k), -(-(n + 1) // k) * k)
    prepared = network.prepare_dataset(rows)
    return lambda: network.classify_rows(prepared)


def case_step(case: str) -> tuple[Callable[[], object], bool]:
    """A case's step, and whether it is an inference step (one loop a repeat)."""
    kind, window, fields, _ = parse_case(case)
    assert kind in ("learn", "forward"), f"{case}: an end-to-end case, indrajala-studies' window_study_timing.py"
    if kind == "learn":
        return learn_step(window, fields), False
    return forward_step(window, fields), True


def control_step() -> Callable[[], object]:
    """A fixed crate matmul, (256, 256) by (256, 256): untouched by window-recurrent changes."""
    import indrajala_math_rust as pa

    rng = np.random.default_rng(0)
    a, b = (pa.Array(rng.uniform(-1, 1, (256, 256)).tolist()) for _ in range(2))
    return lambda: a @ b


def breakdown(case: str, steps: Callable[[str], tuple[Callable[[], object], bool]] = case_step) -> list[str]:
    """A case's step (from steps) once unprofiled and once under cProfile, its components (the module docstring)."""
    step, _ = steps(case)
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


def main(
    description: str | None = __doc__,
    default_cases: list[str] = DEFAULT_CASES,
    steps: Callable[[str], tuple[Callable[[], object], bool]] = case_step,
) -> None:
    """The command line, for this script's cases or (given) another's: window_study_timing.py's."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=default_cases)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--loops", type=int, default=20, help="a learn case's steps a repeat")
    parser.add_argument("--breakdown", nargs="+", metavar="CASE", help="profile these cases; no rows")
    arguments = parser.parse_args()
    if arguments.breakdown:
        for case in arguments.breakdown:
            print("\n".join(breakdown(case, steps)), flush=True)
        return
    timed_steps = [(case.replace("-", " "), *steps(case), False) for case in arguments.cases]
    timed_steps.append(("matmul 256", control_step(), False, True))
    for name, step, inference, control in timed_steps:
        seconds = timed(step, arguments.repeats, 1 if inference else arguments.loops)
        for value in seconds:
            row = {"case": name, "metric": "per step", "value": value, "unit": "s"}
            print(json.dumps({**row, "control": True} if control else row), flush=True)
        print(f"# {name}: median {statistics.median(seconds):.6f} s per step", file=sys.stderr)


if __name__ == "__main__":
    main()
