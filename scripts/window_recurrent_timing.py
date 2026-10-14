"""
A window-recurrent network's training step timed: `learn_batch` of the window-recurrent study's
networks on the Rust backend (`window_recurrent_study.build`, the study's window beside a state of
s features and m codes), a batch of B examples of W windows of k columns, random column tokens and
digits (the time doesn't depend on them). Each case is built from a fixed seed, warmed up, then
timed in repeats of loops steps; each repeat prints a row, its seconds per step. A fixed crate
matmul is timed the same way as the control, a path no window-recurrent change touches.

    python scripts/window_recurrent_timing.py [--cases k1-s4-m16-B32-W12 ...] [--repeats 5] [--loops 20]

It prints the probe contract's lines (scripts/ab.py, `--bench cmd`), one JSON object per row:

    {"case": "k1 s4 m16 B32 W12", "metric": "per step", "value": 0.0123, "unit": "s"}

`build` takes the same arguments before and after the factored state (the factored-state workplan,
stage 4), so ab.py runs this script from the new tree on both sides:

    python scripts/ab.py remote --host jebel run --bench cmd --old OLD --new NEW --order ONNO \\
        -- scripts/window_recurrent_timing.py
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable

# run as `python scripts/window_recurrent_timing.py` from the repo root, which puts scripts/ (not the
# repo root) on sys.path: the root goes last, a fallback, so a PYTHONPATH tree comes first (ab.py
# runs the new tree's script with each side's tree on PYTHONPATH)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# the study's cells: k 1 (E7, E8, C9's) and k 4 (stage 7's), the state and codes of E8b
DEFAULT_CASES = ["k1-s4-m16-B32-W12", "k4-s4-m16-B32-W5"]


def parse_case(case: str) -> tuple[int, int, int, int, int]:
    """'k1-s4-m16-B32-W12' to (k, s, m, B, W)."""
    fields = {part[0]: int(part[1:]) for part in case.split("-")}
    return fields["k"], fields["s"], fields["m"], fields["B"], fields["W"]


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


def learn_step(case: str) -> Callable[[], object]:
    """One case's training step on a fixed batch."""
    from indrajala_ml.studies import window_recurrent_study as wr
    from indrajala_ml.studies import window_tasks as wt

    k, s, m, batch, windows = parse_case(case)
    network = wr.build(wt.ADDITION, k, s, m, seed=0)
    vocabulary = network.layer_specs[0].vocabulary
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, vocabulary, size=(batch, windows * k))
    digits = rng.integers(0, network.class_count, size=(batch, windows * k))
    rows = [(tuple(float(t) for t in x), tuple(int(d) for d in y)) for x, y in zip(tokens, digits, strict=True)]
    return lambda: network.learn_batch(0.001, rows)


def control_step() -> Callable[[], object]:
    """A fixed crate matmul, (256, 256) by (256, 256): untouched by window-recurrent changes."""
    import indrajala_math_rust as pa

    rng = np.random.default_rng(0)
    a, b = (pa.Array(rng.uniform(-1, 1, (256, 256)).tolist()) for _ in range(2))
    return lambda: a @ b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--loops", type=int, default=20)
    arguments = parser.parse_args()
    steps = [(case.replace("-", " "), learn_step(case), False) for case in arguments.cases]
    steps.append(("matmul 256", control_step(), True))
    for name, step, control in steps:
        seconds = timed(step, arguments.repeats, arguments.loops)
        for value in seconds:
            row = {"case": name, "metric": "per step", "value": value, "unit": "s"}
            print(json.dumps({**row, "control": True} if control else row), flush=True)
        print(f"# {name}: median {statistics.median(seconds):.6f} s per step", file=sys.stderr)


if __name__ == "__main__":
    main()
