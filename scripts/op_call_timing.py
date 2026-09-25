"""
Every call of chosen crate functions timed inside a real Rust training run: the conv demo's
`train_once` on its 2000-row MNIST subset, one process per (architecture, trainer, setting,
repeat), the order rotated each repeat. Each call is timed with perf_counter around the crate
function (patched on the module, so every layer's `pa.<name>` call goes through it) and grouped
by its array arguments' shapes, so the layers of one network come out separately. It reports each
group's calls and the median µs per call (min-max over repeats), and the seconds it took in the run,
and the whole run's seconds as the demo times it (the timing wrapper adds about a microsecond a
call; nothing is profiled).

    python scripts/op_call_timing.py [--architectures conv-conv ...] [--trainers ...]
                                     [--op conv_accumulate_gradient_batch ...] [--rust-threads N]
                                     [--kernel-overrides 0:0 1:1000000000 ...] [--repeats 2]
                                     [--json runs.json]

This is the op in place: the calls see the cache and clock state training leaves them (a
back-to-back benchmark loop doesn't), and training's own thread count unless `--rust-threads`
pins one. `--kernel-overrides R:K` runs every cell under `set_kernel_overrides(R, K)` (0:0 is the
default kernels; `matmul_narrow`'s rows per block and `matmul_long_k`'s slab rows), so one build
compares kernel settings. For old/new builds, run it once per build with that build first on
PYTHONPATH, alternating the builds, as docs/optimizations/measurement.md describes.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

# run as `python scripts/op_call_timing.py` from the repo root, which puts scripts/ (not the repo
# root) on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process_runs import interleaved_runs, run_json_worker

from indrajala_ml.demos import demo_conv_rust_vs_vectorized_digit_recognition as demo

DEFAULT_OPS = ["conv_accumulate_gradient_batch"]
DEFAULT_OVERRIDES = "0:0"


def call_key(args: tuple[Any, ...]) -> str:
    """The shapes of a call's array arguments, e.g. "(32, 4608) (18432, 72)"."""
    import indrajala_math_rust as pa

    return " ".join(str(arg.shape) for arg in args if isinstance(arg, pa.Array))


def worker(architecture: str, trainer: str, overrides: str, ops: list[str], rust_threads: int | None) -> dict[str, Any]:
    """{"run_s": the training run's seconds, as the demo times it; op: {shapes: per-call stats}}."""
    import indrajala_math_rust as pa

    if rust_threads is not None:
        pa.set_matmul_threading(rust_threads, 0)
    rows_per_block, k_block = (int(v) for v in overrides.split(":"))
    if (rows_per_block, k_block) != (0, 0):
        pa.set_kernel_overrides(rows_per_block, k_block)

    seconds: dict[str, dict[str, list[float]]] = {op: {} for op in ops}

    def timed(op: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any) -> Any:
            start = time.perf_counter()
            result = fn(*args)
            seconds[op].setdefault(call_key(args), []).append(time.perf_counter() - start)
            return result

        return call

    for op in ops:
        setattr(pa, op, timed(op, getattr(pa, op)))

    datasets = demo.load_datasets()
    side, train_data, _test_data, epochs = datasets[f"MNIST 28x28 ({demo.MNIST_TRAIN_LIMIT} train)"]
    conv_specs = demo.ARCHITECTURES[architecture]
    snapshot = demo.initial_snapshot(side, conv_specs)
    _network, _result, run_s = demo.train_once("rust", trainer, side, conv_specs, snapshot, train_data, epochs)
    return {"run_s": run_s} | {
        op: {
            key: {"calls": len(times), "median_us": statistics.median(times) * 1e6, "total_s": sum(times)}
            for key, times in by_shape.items()
        }
        for op, by_shape in seconds.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--architectures", nargs="+", choices=list(demo.ARCHITECTURES), default=list(demo.ARCHITECTURES)
    )
    parser.add_argument("--trainers", nargs="+", choices=demo.TRAINERS, default=[demo.TRAINERS[1]])
    parser.add_argument("--op", action="append", default=[], help="crate function to time, repeatable")
    parser.add_argument("--rust-threads", type=int, help="set_matmul_threading(N, 0) (default: training's)")
    parser.add_argument(
        "--kernel-overrides", nargs="+", default=[DEFAULT_OVERRIDES], help="R:K settings for set_kernel_overrides"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--json", help="also write every run here")
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    ops: list[str] = args.op or DEFAULT_OPS

    if args.worker:
        architecture, trainer, overrides = args.worker
        print(json.dumps(worker(architecture, trainer, overrides, ops, args.rust_threads)))
        return

    def run_cell(cell: tuple[str, str, str]) -> dict[str, Any]:
        command = [sys.executable, "-B", os.path.abspath(__file__), "--worker", *cell]
        command += [arg for op in ops for arg in ("--op", op)]
        if args.rust_threads is not None:
            command += ["--rust-threads", str(args.rust_threads)]
        return run_json_worker(command)

    cells = [
        (architecture, trainer, overrides)
        for architecture in args.architectures
        for trainer in args.trainers
        for overrides in args.kernel_overrides
    ]
    runs = interleaved_runs(cells, args.repeats, run_cell)

    threads = args.rust_threads or "training's"
    print(
        f"µs per call inside one Rust training run, median (min-max over {args.repeats} processes), threads {threads}\n"
    )
    print("| architecture | trainer | overrides | op | argument shapes | calls | median µs | s in run |")
    print("|---|---|---|---|---|---|---|---|")
    for (architecture, trainer, overrides), cell_runs in runs.items():
        run_s = [run["run_s"] for run in cell_runs]
        print(
            f"| {architecture} | {trainer} | {overrides} | (whole run, s) | | | "
            f"{statistics.median(run_s):.3f} ({min(run_s):.3f}-{max(run_s):.3f}) | |"
        )
        for op in ops:
            for key in cell_runs[0][op]:
                medians = [run[op][key]["median_us"] for run in cell_runs]
                totals = [run[op][key]["total_s"] for run in cell_runs]
                print(
                    f"| {architecture} | {trainer} | {overrides} | {op} | {key} | {cell_runs[0][op][key]['calls']} "
                    f"| {statistics.median(medians):.0f} ({min(medians):.0f}-{max(medians):.0f}) "
                    f"| {min(totals):.3f}-{max(totals):.3f} |"
                )
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"settings": vars(args), "runs": {" / ".join(cell): r for cell, r in runs.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
