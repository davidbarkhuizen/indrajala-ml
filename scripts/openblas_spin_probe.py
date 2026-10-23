"""
Does numpy's OpenBLAS slow a Rust call made right after it? After a threaded BLAS call,
OpenBLAS's worker threads spin for a while (`OPENBLAS_THREAD_TIMEOUT`) before they sleep, and on
the Ryzen laptop a Rust batch op in that window measured 2-5x slow (docs/measurement.md, §7).
This times single Rust calls in one process under three conditions, alternated trial by trial:

- after numpy: a threaded numpy product (512 x 5408 @ 5408 x 32), then the Rust call at once;
- after a sleep: the same numpy product, a sleep past any spin timeout (--sleep-s), then a
  busy-wait on this core (--warm-ms) so its clock is back up, then the Rust call;
- alone: the busy-wait, then the Rust call, with no numpy call in the trial.

The Rust ops are focused_benchmark.py's cases: the conv tail's `downstream_batch` at batch 32 (one
thread by the crate's 8M-flop threshold) and at batch 512 (threaded). It prints one JSON object
per (op, condition), the median µs of one call over --trials, in scripts/ab.py's probe format:

    python scripts/ab.py run --bench cmd --old HEAD --new HEAD -- scripts/openblas_spin_probe.py

Added for the benchmark machine workplan's stage 6.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import numpy as np
from focused_benchmark import case_key, every_case

OPS = ["conv tail 32 x 5408|downstream_batch|32", "conv tail 32 x 5408|downstream_batch|512"]
SEED = 0


def _busy(ms: float) -> None:
    end = time.perf_counter() + ms / 1000
    while time.perf_counter() < end:
        pass


def _timed(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--sleep-s", type=float, default=1.0, help="past OpenBLAS's spin timeout")
    parser.add_argument("--warm-ms", type=float, default=20.0, help="busy-wait so this core's clock is up")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    a, b = rng.uniform(-1, 1, size=(512, 5408)), rng.uniform(-1, 1, size=(5408, 32))
    cases = {case_key(c): c for c in every_case([32, 512], [])}
    for op in OPS:
        rust = cases[op].build("rust")
        rust()  # warm-up
        times: dict[str, list[float]] = {"after numpy": [], "after a sleep": [], "alone": []}
        for _ in range(args.trials):
            _ = a @ b
            times["after numpy"].append(_timed(rust))
            _ = a @ b
            time.sleep(args.sleep_s)
            _busy(args.warm_ms)
            times["after a sleep"].append(_timed(rust))
            _busy(args.warm_ms)
            times["alone"].append(_timed(rust))
        shape, name, batch = op.split("|")
        for condition, values in times.items():
            row = {"case": f"{shape} {name} b{batch} / {condition}", "metric": "one call"}
            print(json.dumps({**row, "value": statistics.median(values), "unit": "µs"}))


if __name__ == "__main__":
    main()
