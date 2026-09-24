"""
The focused per-op benchmark from docs/optimizations/measurement.md, as a tool: every
(case, backend) runs in its own Python process, so numpy's OpenBLAS threads can never spin
while Rust is timed. In each process: one warm-up call, a calibration that sizes a loop to about
`--target-ms`, then `--loops` timed loops. It reports the median microseconds per call (and the
min-max over loops) and the minor page faults per call (`getrusage`, no root needed).

Cases are demo_layer_op_timing's (every dense, conv and max-pool layer op), plus:

- per dense shape and batch, the parts of the two backward batch ops: `bare downstream`
  (`delta_batch @ W`), `bare accumulate` (`delta_batch.T @ X`), `transpose` (`delta_batch.T`),
  `add` (`grad_W + update`) and `sum_axis0` (`delta_batch`);
- `--matmul MxKxN`, a bare `(M, K) @ (K, N)` product, repeatable.

Examples:

    python scripts/focused_benchmark.py --shape 5408 --op downstream_batch --batch 32 --passes 2
    python scripts/focused_benchmark.py --matmul 32x128x1352 --matmul 32x32x5408 --backend rust
    python scripts/focused_benchmark.py --op accumulate --rust-threads 1 --openblas-threads 1
    python scripts/focused_benchmark.py --shape 28x28 --op forward --backend rust --malloc both

`--malloc raised` sets glibc's `MALLOC_TRIM_THRESHOLD_` and `MALLOC_MMAP_THRESHOLD_` to 1e9 in
the timed processes, so freed memory is never handed back to the OS and nothing faults in again;
`--malloc both` times each case under both settings. A time that drops with the faults is paying
for them; one that doesn't is compute or cache traffic.

Passes swap the backend order, so neither backend always runs first. Pure Python is never timed.
"""

import argparse
import json
import os
import resource
import statistics
import sys
import time
from typing import Callable

# run as `python scripts/focused_benchmark.py` from the repo root, which puts scripts/ (not the
# repo root) on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from indrajala_ml.demos.demo_layer_op_timing import (  # noqa: E402
    BACKENDS,
    DENSE_SHAPES,
    SEED,
    Case,
    _backend_array,
    all_cases,
)

from process_runs import run_json_worker  # noqa: E402

PART_OPS = ("bare downstream", "bare accumulate", "transpose", "add", "sum_axis0")


def dense_part_cases(label: str, size: int, input_size: int, batch_sizes) -> list[Case]:
    """The parts of downstream_batch and accumulate_gradient_batch, on the same inputs."""

    def part(op: str, batch: int) -> Callable[[str], Callable[[], object]]:
        def build(backend: str) -> Callable[[], object]:
            rng = np.random.default_rng(SEED)
            W = _backend_array(backend, rng.uniform(-0.3, 0.3, size=(size, input_size)))
            X = _backend_array(backend, rng.uniform(0.0, 1.0, size=(batch, input_size)))
            delta = _backend_array(backend, rng.uniform(-0.1, 0.1, size=(batch, size)))
            grad_w = _backend_array(backend, rng.uniform(-0.1, 0.1, size=(size, input_size)))
            update = _backend_array(backend, rng.uniform(-0.1, 0.1, size=(size, input_size)))
            if op == "bare downstream":
                return lambda: delta @ W
            if op == "bare accumulate":
                delta_t = delta.T.copy() if backend == "numpy" else delta.T
                return lambda: delta_t @ X
            if op == "transpose":
                # numpy's .T is a view; its matmul reads it strided, so the copy is the fair part
                return (lambda: delta.T.copy()) if backend == "numpy" else (lambda: delta.T)
            if op == "add":
                return lambda: grad_w + update
            import indrajala_math_rust as pa

            return (lambda: delta.sum(axis=0)) if backend == "numpy" else (lambda: pa.sum_axis0(delta))

        return build

    return [Case(label, op, batch, part(op, batch)) for op in PART_OPS for batch in batch_sizes]


def matmul_case(spec: str) -> Case:
    m, k, n = (int(v) for v in spec.lower().split("x"))

    def build(backend: str) -> Callable[[], object]:
        rng = np.random.default_rng(SEED)
        a = _backend_array(backend, rng.uniform(-1.0, 1.0, size=(m, k)))
        b = _backend_array(backend, rng.uniform(-1.0, 1.0, size=(k, n)))
        return lambda: a @ b

    return Case(f"matmul {m}x{k}x{n}", "bare matmul", None, build)


def every_case(batch_sizes, matmuls) -> list[Case]:
    cases = all_cases(batch_sizes)
    for label, size, input_size in DENSE_SHAPES:
        cases += dense_part_cases(label, size, input_size, batch_sizes)
    return cases + [matmul_case(spec) for spec in matmuls]


def case_key(case: Case) -> str:
    return f"{case.shape}|{case.op}|{case.batch}"


def measure(fn: Callable[[], object], loops: int, target_s: float) -> dict:
    """Median and range of µs per call over `loops` loops of about `target_s` each, faults per call."""
    fn()
    count, elapsed = 1, 0.0
    while True:
        start = time.perf_counter()
        for _ in range(count):
            fn()
        elapsed = time.perf_counter() - start
        if elapsed >= target_s / 4:
            break
        count *= 2
    count = max(1, round(count * target_s / elapsed))
    per_call, faults = [], 0
    for _ in range(loops):
        faults_before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        start = time.perf_counter()
        for _ in range(count):
            fn()
        per_call.append((time.perf_counter() - start) / count * 1e6)
        faults += resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults_before
    return {
        "calls": count,
        "median_us": statistics.median(per_call),
        "min_us": min(per_call),
        "max_us": max(per_call),
        "faults_per_call": faults / (count * loops),
    }


def worker(args) -> None:
    if args.backend_to_run == "rust" and args.rust_threads is not None:
        import indrajala_math_rust as pa

        pa.set_matmul_threading(args.rust_threads, 0)
    case = next(c for c in every_case(args.batch_sizes, args.matmul) if case_key(c) == args.worker)
    result = measure(case.build(args.backend_to_run), args.loops, args.target_ms / 1000)
    print(json.dumps(result))


# glibc's thresholds at 1e9: freed memory stays in the heap, so it never faults in again.
RAISED_MALLOC_ENV = {"MALLOC_TRIM_THRESHOLD_": "1000000000", "MALLOC_MMAP_THRESHOLD_": "1000000000"}


def run_in_process(case: Case, backend: str, args, malloc: str = "default") -> dict:
    env = dict(os.environ)
    if malloc == "raised":
        env.update(RAISED_MALLOC_ENV)
    if backend == "numpy" and args.openblas_threads is not None:
        env["OPENBLAS_NUM_THREADS"] = str(args.openblas_threads)
    command = [sys.executable, "-B", os.path.abspath(__file__), "--worker", case_key(case), "--backend-to-run", backend]
    command += ["--loops", str(args.loops), "--target-ms", str(args.target_ms)]
    command += ["--batch-sizes", *(str(b) for b in args.batch_sizes)]
    for spec in args.matmul:
        command += ["--matmul", spec]
    if args.rust_threads is not None:
        command += ["--rust-threads", str(args.rust_threads)]
    return run_json_worker(command, env)


def selected(case: Case, args) -> bool:
    if args.matmul and case.op == "bare matmul":
        return True
    if args.only_matmul:
        return False
    return (
        (not args.shape or any(s in case.shape for s in args.shape))
        and (not args.op or any(o in case.op for o in args.op))
        and (not args.batch or case.batch in args.batch)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", action="append", default=[], help="substring of the shape label, repeatable")
    parser.add_argument("--op", action="append", default=[], help="substring of the op name, repeatable")
    parser.add_argument("--batch", action="append", type=int, default=[], help="batch size to keep, repeatable")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 512], help="batch sizes to build")
    parser.add_argument("--matmul", action="append", default=[], help="bare product MxKxN, repeatable")
    parser.add_argument("--only-matmul", action="store_true", help="only the --matmul cases")
    parser.add_argument("--backend", action="append", choices=BACKENDS, default=[])
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--loops", type=int, default=9)
    parser.add_argument("--target-ms", type=float, default=20.0)
    parser.add_argument("--rust-threads", type=int, help="set_matmul_threading(N, 0) before timing Rust")
    parser.add_argument("--openblas-threads", type=int, help="OPENBLAS_NUM_THREADS for the numpy processes")
    parser.add_argument(
        "--malloc",
        choices=["default", "raised", "both"],
        default="default",
        help="glibc allocator thresholds in the timed processes (raised: no trimming, no mmap)",
    )
    parser.add_argument("--json", help="also write every result to this file")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--backend-to-run", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(args)
        return

    backends = args.backend or list(BACKENDS)
    cases = [c for c in every_case(args.batch_sizes, args.matmul) if selected(c, args)]
    mallocs = ["default", "raised"] if args.malloc == "both" else [args.malloc]
    settings = f"rust threads {args.rust_threads or 'default'}, OpenBLAS threads {args.openblas_threads or 'default'}"
    settings += f", malloc {args.malloc}"
    print(f"{len(cases)} cases x {len(backends)} backends x {args.passes} passes, {settings}")
    print(f"median (min-max) µs per call over {args.loops} loops of ~{args.target_ms:g} ms; faults = minor page faults per call")
    header = f"{'pass':>4} {'shape':<20} {'op':<26} {'batch':>5} {'backend':<7} {'malloc':<7} {'median':>9} {'min-max':>17} {'faults':>7}"
    print(header)
    results = []
    for pass_index in range(args.passes):
        order = backends if pass_index % 2 == 0 else list(reversed(backends))
        for case in cases:
            for backend in order:
                for malloc in mallocs if pass_index % 2 == 0 else list(reversed(mallocs)):
                    r = run_in_process(case, backend, args, malloc)
                    results.append(
                        {"pass": pass_index + 1, "shape": case.shape, "op": case.op, "batch": case.batch,
                         "backend": backend, "malloc": malloc, **r}
                    )
                    batch = "-" if case.batch is None else case.batch
                    spread = f"{r['min_us']:.1f}-{r['max_us']:.1f}"
                    print(
                        f"{pass_index + 1:>4} {case.shape:<20} {case.op:<26} {batch:>5} {backend:<7} {malloc:<7} "
                        f"{r['median_us']:>9.1f} {spread:>17} {r['faults_per_call']:>7.1f}",
                        flush=True,
                    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"settings": vars(args), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
