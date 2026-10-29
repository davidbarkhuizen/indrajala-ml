"""
What a threaded call costs the crate (the crate kernels workplan, stage 3, D3): the crate starts OS
threads on every threaded call (`std::thread::scope`), where OpenBLAS keeps a pool. Two
measurements, on whichever machine runs it:

- an empty scoped spawn of 1 to `--spawn-max` threads (`scoped_spawn_cost.rs`, built here with the
  crate's pinned toolchain): the threads' start and join, with no work;
- each `--matmul` product (default the narrow-`k` target, dense 30 x 784's `downstream_batch` at
  b32, `(32, 30) @ (30, 784)`) at each `--threads` count, the crate's threading threshold set to 1
  flop so every count threads (`focused_benchmark --rust-threads N:1`), against numpy at the same
  OpenBLAS thread count (`--openblas-threads N`).

It prints the probe lines `ab.py --bench cmd` reads (`{"case", "metric", "value", "unit"}`, numpy's
marked `"control": true`) and a table. On a machine with a thermal zone it records the peak
temperature, read between measurements.

    python scripts/measurement/thread_cost_probe.py
    python scripts/measurement/thread_cost_probe.py --threads 1 2 4 --matmul 32x30x784 --matmul 512x30x784
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")


class Temperatures:
    def __init__(self) -> None:
        self.peak: float | None = None

    def read(self) -> None:
        if THERMAL.is_file():
            celsius = int(THERMAL.read_text()) / 1000
            self.peak = celsius if self.peak is None else max(self.peak, celsius)


def threads_label(count: int) -> str:
    return f"{count} thread" + ("s" if count > 1 else "")


def emit(case: str, metric: str, value: float, unit: str, control: bool = False) -> None:
    line = {"case": case, "metric": metric, "value": value, "unit": unit}
    print(json.dumps(line | ({"control": True} if control else {})), flush=True)


def spawn_costs(max_threads: int, loops: int, calls: int) -> list[dict[str, float]]:
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "scoped_spawn_cost"
        source = Path(__file__).with_name("scoped_spawn_cost.rs")
        # from rust/, so rustup takes the crate's rust-toolchain.toml
        subprocess.run(
            ["rustc", "-C", "opt-level=3", "--edition", "2024", "-o", str(binary), str(source)],
            cwd=REPO / "rust",
            check=True,
        )
        out = subprocess.run(
            [str(binary), str(max_threads), str(loops), str(calls)], capture_output=True, text=True, check=True
        )
    return [json.loads(line) for line in out.stdout.splitlines()]


def product_times(matmul: str, threads: int, passes: int, loops: int) -> dict[str, float]:
    """Each backend's median µs per call over the passes' medians."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "times.json"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/measurement/focused_benchmark.py"),
                *("--matmul", matmul, "--only-matmul", "--passes", str(passes), "--loops", str(loops)),
                *("--rust-threads", f"{threads}:1", "--openblas-threads", str(threads), "--json", str(out)),
            ],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        results = json.loads(out.read_text())["results"]
    return {
        backend: statistics.median(r["median_us"] for r in results if r["backend"] == backend)
        for backend in ("rust", "numpy")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spawn-max", type=int, default=4, help="empty spawns of 1..N threads")
    parser.add_argument("--spawn-loops", type=int, default=15)
    parser.add_argument("--spawn-calls", type=int, default=2000, help="calls per timed loop")
    parser.add_argument("--matmul", action="append", help="MxKxN, repeatable (default 32x30x784)")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--passes", type=int, default=2, help="focused_benchmark passes per count")
    parser.add_argument("--loops", type=int, default=7, help="focused_benchmark loops per pass")
    args = parser.parse_args()
    temperatures = Temperatures()
    temperatures.read()

    rows: list[tuple[str, float, float | None]] = []
    for cost in spawn_costs(args.spawn_max, args.spawn_loops, args.spawn_calls):
        case = f"empty scoped spawn, {threads_label(int(cost['threads']))}"
        emit(case, "us per call", cost["median_us"], "us")
        rows.append((case, cost["median_us"], None))
    temperatures.read()
    for matmul in args.matmul or ["32x30x784"]:
        for threads in args.threads:
            times = product_times(matmul, threads, args.passes, args.loops)
            temperatures.read()
            case = f"matmul {matmul}, {threads_label(threads)}"
            emit(case, "us per call", times["rust"], "us")
            emit(case, "us per call", times["numpy"], "us", control=True)
            rows.append((case, times["rust"], times["numpy"]))
    if temperatures.peak is not None:
        emit("the probe", "peak temperature", temperatures.peak, "C")

    print(f"\n{'case':42} {'rust µs':>9} {'numpy µs':>9}")
    for case, rust, numpy in rows:
        print(f"{case:42} {rust:9.1f} {'' if numpy is None else f'{numpy:9.1f}':>9}")
    if temperatures.peak is not None:
        print(f"peak {temperatures.peak:.1f} C")


if __name__ == "__main__":
    main()
