"""
What threads cost and what they buy (the crate performance workplan, stage 8). One machine at a
time, three questions, each a subcommand; D6 says the last decides, the first two say where to look.

- `calls`: one call handed to threads, its work fixed (`thread_call_cost.rs`, built here with the
  crate's pinned toolchain): `std::thread::scope`, as the crate does today, against a prototype of
  D3's parked pool at each `--spin-us`, at each thread count, warm (back-to-back) and cold (`--idle-ms`
  of sleep before each call, timing only the call: the workers parked, the cores cooled).
- `scaling`: threads x shape. Each product or layer op, the crate's threshold at 1 flop so every
  count threads, warm and cold, against numpy at the same OpenBLAS count (`focused_benchmark`).
  `--matmul MxKxN` is a bare `a @ b`; `--op "SHAPE|OP|BATCH"` a layer op, its fields as `focused_benchmark`'s
  `--shape`, `--op` and `--batch`. A layer's forward is how `matmul_nt` is reached.
- `epochs`: N processes x T threads of a backdrop on one host, so training's own thread count and
  oversubscription are one sweep. Each process runs the backdrop once, Rust only, at
  `OPENBLAS_NUM_THREADS=1` as the studies run, all N started together; T 0 is training's default.
  Arguments after `--` go to the backdrop. **Plan the memory first:** N copies of the backdrop's peak
  must fit (`docs/measurement.md`, §2), and `d` 512 is large.

Every process is one cell, in the rotating order `process_runs.interleaved_runs` gives, since
all-core work carries over from one setting to the next through the power budget
(docs/measurement.md, Gotchas). It prints the probe lines `ab.py --bench cmd` reads (numpy's marked
`"control": true`), then a table of medians, each with its gain over its group's first row. On a
machine with a thermal zone it records the peak temperature, read between cells.

    python scripts/measurement/thread_cost_probe.py calls
    python scripts/measurement/thread_cost_probe.py scaling --threads 1 2 4 --matmul 512x30x784
    python scripts/measurement/thread_cost_probe.py epochs --backdrop sequence --threads 0 1 8 --processes 1 8 -- --windows 256
    python scripts/measurement/ab.py remote --host jebel run --bench cmd --old main --new main --order ONNO \\
        -- scripts/measurement/thread_cost_probe.py scaling

The backdrops take the setting themselves too (`prepared_dataset_timing`'s is `--rust-threading`), for
an old-against-new A/B at one thread count:

    python scripts/measurement/ab.py remote --host jebel run --bench cmd --old-crate OLD --new-crate NEW \\
        -- scripts/measurement/sequence_training_timing.py --backends rust --rust-threads 4:1000000
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from process_runs import interleaved_runs, run_json_worker

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")

# what the census (the workplan's K25) found threaded or nearly so: `d` 512's three products, K7's
# long `k`, the shape that lost when it threaded (5.5M flops), and the narrow-`k` spawn case
DEFAULT_MATMULS = ("2048x2048x512", "2048x512x2048", "512x2048x2048", "64x256x1024", "32x128x1352", "32x30x784")
DEFAULT_OPS = (
    "dense 30 x 784|forward_batch|512",  # matmul_nt, 12M flops
    "conv tail 32 x 5408|downstream_batch|512",  # 88.6M flops
    "conv 26x26x8, 8 ch|downstream_batch|512",  # conv's split over examples
)


class Temperatures:
    def __init__(self) -> None:
        self.peak: float | None = None

    def read(self) -> None:
        if THERMAL.is_file():
            celsius = int(THERMAL.read_text()) / 1000
            self.peak = celsius if self.peak is None else max(self.peak, celsius)


@dataclass
class Table:
    """Each group's rows in the order they were first seen, medians over the repeats."""

    # (group, row): the repeats' values, and the work one value covers (N processes' epochs each)
    values: dict[tuple[str, str], list[float]]
    work: dict[tuple[str, str], float]

    def add(self, group: str, row: str, value: float, work: float = 1.0) -> None:
        self.values.setdefault((group, row), []).append(value)
        self.work[group, row] = work

    def lines(self) -> list[str]:
        base: dict[str, float] = {}
        lines = [f"{'group':58} {'row':28} {'median':>12} {'gain':>6}"]
        for (group, row), values in sorted(self.values.items(), key=lambda item: item[0][0]):
            rate = self.work[group, row] / statistics.median(values)
            gain = rate / base.setdefault(group, rate)
            lines.append(f"{group:58} {row:28} {statistics.median(values):12.6g} {gain:6.2f}")
        return lines


def threads_label(count: int) -> str:
    return "default threads" if count == 0 else f"{count} thread" + ("s" if count > 1 else "")


def warmth(idle_ms: float) -> str:
    return f"cold ({idle_ms:g} ms idle)" if idle_ms else "warm"


def emit(case: str, metric: str, value: float, unit: str, control: bool = False) -> None:
    line = {"case": case, "metric": metric, "value": value, "unit": unit}
    print(json.dumps(line | ({"control": True} if control else {})), flush=True)


# ---- calls


def build_call_cost(directory: str) -> Path:
    binary = Path(directory) / "thread_call_cost"
    # from rust/, so rustup takes the crate's rust-toolchain.toml
    subprocess.run(
        ["rustc", "-C", "opt-level=3", "--edition", "2024", "-o", str(binary), str(HERE / "thread_call_cost.rs")],
        cwd=REPO / "rust",
        check=True,
    )
    return binary


def run_calls(args: argparse.Namespace, temperatures: Temperatures, table: Table) -> None:
    mechanisms = [("scoped", 0), *(("pool", spin) for spin in args.spin_us)]
    cells = [
        (mechanism, spin, threads, idle, work)
        for mechanism, spin in mechanisms
        for threads in args.threads
        for idle in args.idle_ms
        for work in args.work_k
    ]

    def run(cell: tuple[str, int, int, float, int]) -> dict[str, Any]:
        mechanism, spin, threads, idle, work = cell
        numbers = (threads, round(idle * 1000), work, spin, args.loops, args.target_ms)
        cost = run_json_worker([str(binary), mechanism, *(str(number) for number in numbers)])
        temperatures.read()
        name = mechanism if mechanism == "scoped" else f"pool, spin {spin} µs"
        group = f"{work}k steps, {warmth(idle)}"
        row = f"{name}, {threads_label(threads)}"
        emit(f"{group}, {row}", "per call", cost["median_us"], "µs")
        table.add(group, row, cost["median_us"])
        return cost

    with tempfile.TemporaryDirectory() as directory:
        binary = build_call_cost(directory)
        interleaved_runs(cells, args.repeats, run)


# ---- scaling


def focused_arguments(spec: str) -> list[str]:
    """`focused_benchmark`'s selection for a sweep's product (MxKxN) or layer op (SHAPE|OP|BATCH)."""
    if "|" not in spec:
        return ["--matmul", spec, "--only-matmul"]
    shape, op, batch = spec.split("|")
    return ["--shape", shape, "--op", op, "--batch", batch, "--batch-sizes", batch]


def focused_call(spec: str, backend: str, threads: int, idle_ms: float, loops: int) -> float:
    """The median µs per call of one focused_benchmark process."""
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "times.json"
        subprocess.run(
            [
                sys.executable,
                str(HERE / "focused_benchmark.py"),
                *focused_arguments(spec),
                *("--backend", backend, "--passes", "1", "--loops", str(loops)),
                *("--rust-threads", f"{threads}:1", "--openblas-threads", str(threads)),
                *("--idle-ms", str(idle_ms), "--json", str(out)),
            ],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        results = json.loads(out.read_text())["results"]
    assert len(results) == 1, f"{spec}: selects {len(results)} cases"
    return results[0]["median_us"]


def run_scaling(args: argparse.Namespace, temperatures: Temperatures, table: Table) -> None:
    chosen: list[str] = [*(args.matmul or []), *(args.op or [])]
    specs = chosen or [*DEFAULT_MATMULS, *DEFAULT_OPS]
    cells = [
        (spec, backend, threads, idle)
        for spec in specs
        for backend in args.backends
        for threads in args.threads
        for idle in args.idle_ms
    ]

    def run(cell: tuple[str, str, int, float]) -> dict[str, Any]:
        spec, backend, threads, idle = cell
        micros = focused_call(spec, backend, threads, idle, args.loops)
        temperatures.read()
        group = f"{spec}, {warmth(idle)} / {backend}"
        emit(
            f"{spec}, {threads_label(threads)}, {warmth(idle)} / {backend}",
            "per call",
            micros,
            "µs",
            backend == "numpy",
        )
        table.add(group, threads_label(threads), micros)
        return {}

    interleaved_runs(cells, args.repeats, run)


# ---- epochs


@dataclass(frozen=True)
class Backdrop:
    script: str
    arguments: tuple[str, ...]
    threads_flag: str = "--rust-threads"
    out_flag: str | None = None  # None: its probe lines on stdout


BACKDROPS = {
    "sequence": Backdrop("sequence_training_timing.py", ("--backends", "rust", "--repeats", "1")),
    "attention": Backdrop("attention_timing.py", ("--backends", "rust", "--repeats", "1", "--no-memory")),
    "window": Backdrop("window_recurrent_timing.py", ("--repeats", "1")),
    "prepared": Backdrop(
        "prepared_dataset_timing.py", ("time", "--backends", "rust", "--repeats", "1"), "--rust-threading", "--out"
    ),
}


def backdrop_rows(backdrop: Backdrop, stdout: str, out: Path) -> list[tuple[str, str, float, str, bool]]:
    """(case, metric, value, unit, control) of one process, its probe lines or its --out file."""
    if backdrop.out_flag:
        runs: dict[str, list[dict[str, float]]] = json.loads(out.read_text())
        return [
            (case, metric, value, "s", False)
            for case, case_runs in runs.items()
            for run in case_runs
            for metric, value in run.items()
        ]
    rows: list[tuple[str, str, float, str, bool]] = []
    for line in stdout.splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            if row["unit"] == "s":
                rows.append((row["case"], row["metric"], row["value"], "s", bool(row.get("control"))))
    return rows


def run_backdrop(backdrop: Backdrop, setting: str, processes: int, extra: Sequence[str]) -> list[Any]:
    """Every process's rows, the N started together, each through a file so no pipe fills."""
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory() as directory:
        started: list[tuple[subprocess.Popen[bytes], Path, Path, Any]] = []
        for index in range(processes):
            stdout_path, out = Path(directory) / f"{index}.stdout", Path(directory) / f"{index}.json"
            command = [sys.executable, "-B", str(HERE / backdrop.script), *backdrop.arguments, *extra]
            command += [backdrop.threads_flag, setting, *((backdrop.out_flag, str(out)) if backdrop.out_flag else ())]
            stdout = stdout_path.open("wb")
            started.append((subprocess.Popen(command, env=environment, stdout=stdout), stdout_path, out, stdout))
        rows: list[Any] = []
        for process, stdout_path, out, stdout in started:
            code = process.wait()
            stdout.close()
            if code != 0:
                raise SystemExit(f"{backdrop.script} at {setting} exited {code}")
            rows += backdrop_rows(backdrop, stdout_path.read_text(), out)
    return rows


def run_epochs(args: argparse.Namespace, temperatures: Temperatures, table: Table) -> None:
    backdrop = BACKDROPS[args.backdrop]
    extra = args.backdrop_args[1:] if args.backdrop_args[:1] == ["--"] else args.backdrop_args
    cells = [(processes, threads) for processes in args.processes for threads in args.threads]

    def run(cell: tuple[int, int]) -> dict[str, Any]:
        processes, threads = cell
        setting = f"{threads}:{args.threshold}"
        rows = run_backdrop(backdrop, setting, processes, extra)
        temperatures.read()
        for case, metric, value, unit, control in rows:
            emit(f"{processes} x {threads_label(threads)}, {case}", metric, value, unit, control)
            table.add(f"{case}, {metric}", f"{processes} x {threads_label(threads)}", value, processes)
        return {}

    interleaved_runs(cells, args.repeats, run)


# ---- the command line


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)

    calls = commands.add_parser("calls", help="one call handed to threads: scoped against a pool")
    calls.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    calls.add_argument("--idle-ms", type=float, nargs="+", default=[0, 20], help="0 is warm")
    calls.add_argument("--work-k", type=int, nargs="+", default=[0, 100, 1000], help="thousands of steps a call")
    calls.add_argument("--spin-us", type=int, nargs="+", default=[0, 50], help="the pool's spins before it parks")
    calls.add_argument("--repeats", type=int, default=2)
    calls.add_argument("--loops", type=int, default=9)
    calls.add_argument("--target-ms", type=int, default=20)

    scaling = commands.add_parser("scaling", help="threads x shape, against OpenBLAS")
    scaling.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    scaling.add_argument("--idle-ms", type=float, nargs="+", default=[0, 20], help="0 is warm")
    scaling.add_argument("--matmul", action="append", help="MxKxN, repeatable")
    scaling.add_argument("--op", action="append", help="SHAPE|OP|BATCH, repeatable")
    scaling.add_argument("--backends", nargs="+", default=["rust", "numpy"], choices=["rust", "numpy"])
    scaling.add_argument("--repeats", type=int, default=2)
    scaling.add_argument("--loops", type=int, default=3)

    epochs = commands.add_parser("epochs", help="N processes x T threads of a backdrop")
    epochs.add_argument("--backdrop", required=True, choices=sorted(BACKDROPS))
    epochs.add_argument("--threads", type=int, nargs="+", default=[0], help="0 is training's default")
    epochs.add_argument("--threshold", type=int, default=0, help="flops; 0 is training's default")
    epochs.add_argument("--processes", type=int, nargs="+", default=[1])
    epochs.add_argument("--repeats", type=int, default=2)
    epochs.add_argument("backdrop_args", nargs=argparse.REMAINDER, help="-- then the backdrop's own arguments")
    return top


def main() -> None:
    args = parser().parse_args()
    temperatures = Temperatures()
    temperatures.read()
    table = Table({}, {})
    {"calls": run_calls, "scaling": run_scaling, "epochs": run_epochs}[args.command](args, temperatures, table)
    if temperatures.peak is not None:
        emit("the probe", "peak temperature", temperatures.peak, "C")
    print("\n" + "\n".join(table.lines()))
    if temperatures.peak is not None:
        print(f"peak {temperatures.peak:.1f} C")


if __name__ == "__main__":
    main()
