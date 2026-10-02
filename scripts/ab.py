"""
Timing A/Bs between two commits, run and reported by the protocol in
docs/measurement.md (the A/B harness workplan in docs/next-steps.md has the design):

    python scripts/ab.py run --bench prepared_dataset_timing [--old main] [--new HEAD] [--order ONNONO]
                             [--name NAME] [--control-backend numpy] [--script-from new]
                             [--allow-profile-change] [-- <benchmark arguments>]
    python scripts/ab.py run --bench cmd [...] -- probe.py [probe arguments]
    python scripts/ab.py run --bench prepared_dataset_timing --old main --new main --old-crate 638ff13
                             --new-crate 750d83a --control-backend numpy
    python scripts/ab.py status [RUN]
    python scripts/ab.py extend [RUN] --order NO
    python scripts/ab.py report [RUN] [--brief] [--md FILE]
    python scripts/ab.py clean [--worktrees] [--wheels]

Each side is a commit, checked out once as a detached worktree under ~/code/ab-worktrees/<sha7>.
A run lives in ~/code/ab-runs/<YYYY-MM-DD>-<name>/ (manifest.json, progress.jsonl, and each pass's
raw output and logs), which is also the neutral working directory every pass runs from, with a
`data` symlink to this checkout's data/. A pass runs the benchmark once, in its own process tree,
with only its side's tree on PYTHONPATH; a probe first checks that the trainer
(indrajala_ml.training.train, or indrajala_ml.train in a tree from before the source layout) comes
from that tree and that the crate extension's hash is the run's. `run` does a smoke run of each
side with the benchmark's smallest settings before the passes. RUN defaults to the most recent run.

Output is bounded: raw data goes to files only, `run` and `extend` print a line when they start
and one when they finish (or the failing step's last 20 lines of stderr, exiting 1), and
`report --brief` prints at most 15 lines. `report --md FILE` writes the full table and a protocol
paragraph for a PR body.

The report pools every complete pass of a side: per (metric, case), the median and min-max over
all runs, Δ median, and each pass's own median. A row is *consistent* when every per-pass median
of one side lies beyond every one of the other (2+ passes a side) and the gap between the sides
is wider than each side's own spread of per-pass medians; otherwise it is within noise. A pass
is *shifted* when its rows, the controls included, sit 5% or more from their side's pooled
medians in the same direction; the report names the `extend` order that balances shifted passes.

Benchmarks: prepared_dataset_timing (control: `prepare`, plus the other backend's rows with
--control-backend), and cmd, a probe that prints one JSON object per line to stdout:
{"case": ..., "metric": ..., "value": ..., "unit": ...} (optionally "control": true).
Crate A/Bs: when the sides' crate commits differ (each tree's rust/ submodule, or --old-crate /
--new-crate, which alone make a crate-only A/B), each crate commit is built once into a release
wheel under ~/code/ab-runs/wheels/<sha>/ (from a `git archive` of ~/code/indrajala-math-rust, with
the toolchain it pins) and installed with `pip install --target` into a site directory there,
which goes on PYTHONPATH after the tree; each pass checks the extension's hash. The venv is never
touched. When they are equal, both sides use the venv's extension.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT = Path(os.environ.get("AB_RUNS_ROOT", Path.home() / "code/ab-runs"))
WORKTREES_ROOT = Path(os.environ.get("AB_WORKTREES_ROOT", Path.home() / "code/ab-worktrees"))
CRATE_REPO = Path(os.environ.get("AB_CRATE_REPO", Path.home() / "code/indrajala-math-rust"))
CARGO_BIN = Path.home() / ".cargo/bin"
PROFILE_REFERENCE = "docs/machine_profiles/ryzen7-3700u.json"
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
DIRTY_PATHS = ("indrajala_ml", "scripts", "rust")
SHIFT = 0.05  # a pass this far from its side's pooled medians is shifted
BUSY_PERCENT = 10.0  # a process above this share of one CPU is recorded as busy
HIGH_LOAD = 1.5  # a 1-minute load average above this is reported
BRIEF_LINES = 15
STDERR_TAIL = 20
CLEAN_DAYS = 14

# printed as JSON by a process in a pass's environment: where the trainer and the crate
# extension resolve from, and the extension's hash. argv[1] is the side's tree
PROVENANCE_PROBE = r"""
import hashlib, importlib, importlib.util, json, pathlib, sys
# the trainer, in the tree's own source layout (docs/source-layout-workplan.md, D4); chosen by
# file, not by trying imports: indrajala_ml is a namespace package, so in a tree from before the
# layout, indrajala_ml.training resolves to the editable install's checkout
tree = pathlib.Path(sys.argv[1])
moved = (tree / "indrajala_ml/training/train.py").exists()
train = importlib.import_module("indrajala_ml.training.train" if moved else "indrajala_ml.train")
result = {"train": train.__file__, "extension": None, "sha256": None}
spec = importlib.util.find_spec("indrajala_math_rust")
if spec is not None and spec.origin is not None:
    origin = pathlib.Path(spec.origin)
    files = sorted(origin.parent.glob("*.so")) if origin.name == "__init__.py" else [origin]
    if files:
        result["extension"] = str(files[0])
        result["sha256"] = hashlib.sha256(files[0].read_bytes()).hexdigest()
print(json.dumps(result))
"""


class AbError(Exception):
    """A step failed: its message, and the stderr lines to show (at most STDERR_TAIL)."""

    def __init__(self, message: str, tail: list[str] | None = None) -> None:
        super().__init__(message)
        self.tail = (tail or [])[-STDERR_TAIL:]


@dataclass(frozen=True)
class Row:
    case: str
    metric: str
    value: float
    unit: str
    control: bool = False


class Adapter(Protocol):
    """One timing script: how to run it, its smallest settings, and how to read its output."""

    name: str
    default_args: tuple[str, ...]
    no_control: str  # the brief report's line when no row is a control

    def smoke_args(self, args: list[str]) -> list[str]: ...

    def command(self, script_tree: Path, args: list[str], out: Path) -> list[str]:
        """The arguments after `python`: the script, from script_tree, writing its output to out."""
        ...

    def rows(self, out: Path, stdout: Path, control_backend: str | None) -> list[Row]: ...


class PreparedDatasetTiming:
    name: str = "prepared_dataset_timing"
    default_args: tuple[str, ...] = ("--repeats", "5")
    no_control: str = "controls: none"

    def smoke_args(self, args: list[str]) -> list[str]:
        return ["--configs", "conv B=32", "--repeats", "1"]

    def command(self, script_tree: Path, args: list[str], out: Path) -> list[str]:
        return [str(script_tree / "scripts/prepared_dataset_timing.py"), "time", *args, "--out", str(out)]

    def rows(self, out: Path, stdout: Path, control_backend: str | None) -> list[Row]:
        runs: dict[str, list[dict[str, float]]] = json.loads(out.read_text())
        return [
            Row(case, metric, value, "s", metric == "prepare" or _is_backend(case, control_backend))
            for case, case_runs in runs.items()
            for run in case_runs
            for metric, value in run.items()
        ]


class CommandProbe:
    """A probe script (the probe contract): its first argument is the script, the rest its own."""

    name: str = "cmd"
    default_args: tuple[str, ...] = ()
    no_control: str = 'controls: none (a probe row can carry "control": true)'

    def smoke_args(self, args: list[str]) -> list[str]:
        return args

    def command(self, script_tree: Path, args: list[str], out: Path) -> list[str]:
        if not args:
            raise AbError("--bench cmd needs the probe after --: ab.py run --bench cmd -- probe.py [args]")
        return [str(_probe_path(script_tree, args[0])), *args[1:]]

    def rows(self, out: Path, stdout: Path, control_backend: str | None) -> list[Row]:
        rows: list[Row] = []
        for line in stdout.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and {"case", "metric", "value", "unit"} <= item.keys():
                fields: dict[str, Any] = item  # pyright: ignore[reportUnknownVariableType]
                case = str(fields["case"])
                control = bool(fields.get("control", False)) or _is_backend(case, control_backend)
                rows.append(Row(case, str(fields["metric"]), float(fields["value"]), str(fields["unit"]), control))
        return rows


class FocusedBenchmark:
    """Per-op µs per call (the median over its loops); a case is shape, op, batch, backend and a
    raised malloc setting, and each of the script's --passes is a run."""

    name: str = "focused_benchmark"
    default_args: tuple[str, ...] = ()
    no_control: str = "controls: none (--control-backend reads the other backend's rows as controls)"

    def smoke_args(self, args: list[str]) -> list[str]:
        return ["--shape", "dense 10 x 30", "--op", "forward", "--batch-sizes", "32", "--passes", "1", "--loops", "1"]

    def command(self, script_tree: Path, args: list[str], out: Path) -> list[str]:
        return [str(script_tree / "scripts/focused_benchmark.py"), *args, "--json", str(out)]

    def rows(self, out: Path, stdout: Path, control_backend: str | None) -> list[Row]:
        rows: list[Row] = []
        for result in json.loads(out.read_text())["results"]:
            case = f"{result['shape']} {result['op']}" + (f" b{result['batch']}" if result["batch"] else "")
            case += " (malloc raised)" if result["malloc"] == "raised" else ""
            case += f" / {result['backend']}"
            rows.append(Row(case, "per call", result["median_us"], "µs", _is_backend(case, control_backend)))
        return rows


class EpochOpProfile:
    """Rust seconds per crate op in a profiled conv-demo training run (Rust only: no control)."""

    name: str = "epoch_op_profile"
    default_args: tuple[str, ...] = ()
    no_control: str = "controls: none (Rust only; read the profiled total's share)"

    def smoke_args(self, args: list[str]) -> list[str]:
        return ["--architectures", "conv", "--trainers", "mini-batch (32)", "--repeats", "1"]

    def command(self, script_tree: Path, args: list[str], out: Path) -> list[str]:
        return [str(script_tree / "scripts/epoch_op_profile.py"), *args, "--out", str(out)]

    def rows(self, out: Path, stdout: Path, control_backend: str | None) -> list[Row]:
        rows: list[Row] = []
        for config, runs in json.loads(out.read_text())["runs"].items():
            for run in runs:
                rows.append(Row(f"{config} (profiled total)", "seconds in the run", run["total"], "s"))
                rows += [
                    Row(f"{config} {op}", "seconds in the run", seconds, "s") for op, (seconds, _) in run["ops"].items()
                ]
        return rows


class AccuracyPassTiming:
    """Accuracy-pass and epoch seconds per network and backend (the mismatch counts are not timed)."""

    name: str = "accuracy_pass_timing"
    default_args: tuple[str, ...] = ("--repeats", "5")
    no_control: str = "controls: none (--control-backend reads the other backend's rows as controls)"

    def smoke_args(self, args: list[str]) -> list[str]:
        return ["--networks", "conv", "--repeats", "1"]

    def command(self, script_tree: Path, args: list[str], out: Path) -> list[str]:
        return [str(script_tree / "scripts/accuracy_pass_timing.py"), "time", *args, "--out", str(out)]

    def rows(self, out: Path, stdout: Path, control_backend: str | None) -> list[Row]:
        runs: dict[str, list[dict[str, float]]] = json.loads(out.read_text())
        return [
            Row(case, metric, value, "s", _is_backend(case, control_backend))
            for case, case_runs in runs.items()
            for run in case_runs
            for metric, value in run.items()
            if not metric.startswith("mismatches")
        ]


class OpCallTiming:
    """µs per call of chosen crate functions inside a Rust training run, by argument shapes, and the
    run's seconds (Rust only: no control)."""

    name: str = "op_call_timing"
    default_args: tuple[str, ...] = ()
    no_control: str = "controls: none (Rust only; the whole run's seconds are the context)"

    def smoke_args(self, args: list[str]) -> list[str]:
        return ["--architectures", "conv", "--repeats", "1"]

    def command(self, script_tree: Path, args: list[str], out: Path) -> list[str]:
        return [str(script_tree / "scripts/op_call_timing.py"), *args, "--json", str(out)]

    def rows(self, out: Path, stdout: Path, control_backend: str | None) -> list[Row]:
        rows: list[Row] = []
        for cell, runs in json.loads(out.read_text())["runs"].items():
            for run in runs:
                rows.append(Row(cell, "whole run", run["run_s"], "s"))
                for op, by_shapes in run.items():
                    if op != "run_s":
                        rows += [
                            Row(f"{cell} / {op} {shapes}", "per call", stats["median_us"], "µs")
                            for shapes, stats in by_shapes.items()
                        ]
        return rows


class BatchSizeTiming:
    """Dense full-MNIST epoch parts per batch size and backend (`time` mode)."""

    name: str = "batch_size_timing"
    default_args: tuple[str, ...] = ("--repeats", "5")
    no_control: str = "controls: none (--control-backend reads the other backend's rows as controls)"

    def smoke_args(self, args: list[str]) -> list[str]:
        return ["--batch-sizes", "1024", "--repeats", "1"]

    def command(self, script_tree: Path, args: list[str], out: Path) -> list[str]:
        return [str(script_tree / "scripts/batch_size_timing.py"), "time", *args, "--out", str(out)]

    def rows(self, out: Path, stdout: Path, control_backend: str | None) -> list[Row]:
        rows: list[Row] = []
        for key, runs in json.loads(out.read_text()).items():
            backend, batch_size = key.split()
            case = f"B={batch_size} / {backend}"
            rows += [
                Row(case, metric, value, "s", _is_backend(case, control_backend))
                for run in runs
                for metric, value in run.items()
            ]
        return rows


ADAPTERS: dict[str, Adapter] = {
    adapter.name: adapter
    for adapter in (
        PreparedDatasetTiming(),
        FocusedBenchmark(),
        EpochOpProfile(),
        AccuracyPassTiming(),
        OpCallTiming(),
        BatchSizeTiming(),
        CommandProbe(),
    )
}


def _is_backend(case: str, backend: str | None) -> bool:
    return backend is not None and case.endswith(f"/ {backend}")


def _probe_path(script_tree: Path, probe: str) -> Path:
    """An absolute probe as given; a relative one from the script tree, else from the cwd."""
    path = Path(probe)
    if path.is_absolute():
        return path
    return script_tree / path if (script_tree / path).exists() else Path.cwd() / path


# ---- small helpers


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True, text=True)
    if result.returncode:
        raise AbError(f"git {' '.join(args)} failed", result.stderr.splitlines())
    return result.stdout.strip()


def _local_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC).astimezone()


def _now() -> str:
    return _local_now().isoformat(timespec="seconds")


def _clock(iso: str) -> str:
    return iso[11:16]


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    tmp.replace(path)


def _append_progress(run_dir: Path, event: dict[str, Any]) -> None:
    with open(run_dir / "progress.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")


def _tail(path: Path) -> list[str]:
    return path.read_text(errors="replace").splitlines()[-STDERR_TAIL:] if path.exists() else []


def parse_order(order: str) -> list[str]:
    """ONNO -> ["old", "new", "new", "old"]."""
    if not order or set(order.upper()) - {"O", "N"}:
        raise AbError(f"--order takes O and N only, e.g. ONNONO, not {order!r}")
    return ["old" if letter == "O" else "new" for letter in order.upper()]


def _order_text(sides: list[str]) -> str:
    return "".join("O" if side == "old" else "N" for side in sides)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def busy_processes(interval: float = 0.5) -> list[str]:
    """Processes above BUSY_PERCENT of one CPU over interval seconds, busiest first."""

    def cpu_ticks() -> dict[int, tuple[str, int]]:
        ticks: dict[int, tuple[str, int]] = {}
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                text = stat.read_text()
            except OSError:
                continue
            name = text[text.index("(") + 1 : text.rindex(")")]
            fields = text[text.rindex(")") + 2 :].split()
            ticks[int(stat.parent.name)] = (name, int(fields[11]) + int(fields[12]))
        return ticks

    if not Path("/proc/self/stat").exists():
        return []
    before = cpu_ticks()
    time.sleep(interval)
    after = cpu_ticks()
    per_second = os.sysconf("SC_CLK_TCK")
    busy: list[tuple[float, str]] = []
    for pid, (name, ticks) in after.items():
        if pid in before and pid != os.getpid():
            percent = 100 * (ticks - before[pid][1]) / per_second / interval
            if percent > BUSY_PERCENT:
                busy.append((percent, f"{name}({pid}) {percent:.0f}%"))
    return [text for _, text in sorted(busy, reverse=True)]


# ---- runs on disk


def _manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text())


def _run_dirs() -> list[Path]:
    if not RUNS_ROOT.is_dir():
        return []
    return sorted(
        (d for d in RUNS_ROOT.iterdir() if (d / "manifest.json").is_file()),
        key=lambda d: _manifest(d)["created"],
    )


def find_run(run: str | None) -> Path:
    """A run by path, by directory name under RUNS_ROOT, by name suffix, or the most recent."""
    if run is None:
        runs = _run_dirs()
        if not runs:
            raise AbError(f"no runs under {RUNS_ROOT}")
        return runs[-1]
    for candidate in (Path(run), RUNS_ROOT / run):
        if (candidate / "manifest.json").is_file():
            return candidate
    matches = [d for d in _run_dirs() if d.name.endswith(run)]
    if len(matches) != 1:
        raise AbError(f"no single run matches {run!r} under {RUNS_ROOT}")
    return matches[0]


def _new_run_dir(name: str) -> Path:
    base = RUNS_ROOT / f"{_local_now().date().isoformat()}-{name.replace('/', '-')}"
    run_dir, n = base, 1
    while run_dir.exists():
        n += 1
        run_dir = Path(f"{base}-{n}")
    run_dir.mkdir(parents=True)
    return run_dir


def ensure_worktree(repo: Path, commit: str) -> Path:
    """A detached worktree of commit under WORKTREES_ROOT, created once and reused."""
    tree = WORKTREES_ROOT / commit[:7]
    if tree.exists():
        if _git(tree, "rev-parse", "HEAD") != commit:
            raise AbError(f"{tree} exists but isn't at {commit[:7]}; remove it or run ab.py clean --worktrees")
        return tree
    WORKTREES_ROOT.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "--detach", str(tree), commit)
    return tree


def _crate_commit(repo: Path, commit: str) -> str | None:
    listing = _git(repo, "ls-tree", commit, "rust")
    return listing.split()[2] if listing else None


# ---- crate builds (stage 3): one wheel per crate commit, installed into a site directory


def _wheels_root() -> Path:
    return RUNS_ROOT / "wheels"


def _wheel_dir(crate: str) -> Path:
    return _wheels_root() / crate


BUILT: list[str] = []  # crate commits built by this process, for the machine line


def extract_crate(crate: str, source: Path) -> None:
    """Crate commit `crate`'s files in a fresh `source` directory, stamped with the extraction time.

    `git archive` stamps every file with the commit time, and cargo decides freshness in the shared
    target directory by modification time: a commit made before the last build there looked up to
    date and wasn't compiled, so its wheel held the previous commit's extension (the stage 4c A/B)."""
    shutil.rmtree(source, ignore_errors=True)
    source.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(CRATE_REPO), "archive", crate], check=False, capture_output=True)
    if archive.returncode:
        raise AbError(f"crate {crate[:7]}: git archive failed", archive.stderr.decode().splitlines())
    subprocess.run(["tar", "-x", "--touch", "-C", str(source)], input=archive.stdout, check=True)


CRATE_BUILD_INPUTS = ("src", "build.rs", "Cargo.toml", "Cargo.lock", "pyproject.toml", "rust-toolchain.toml")


def check_crate_builds(crates: dict[str, str], shas: dict[str, str]) -> None:
    """Raises when the sides' crate commits differ in what the extension is built from but their
    extensions are the same file: one side's build is stale, and the A/B would time one crate twice."""
    if shas["old"] != shas["new"]:
        return
    changed = _git(CRATE_REPO, "diff", "--name-only", crates["old"], crates["new"], "--", *CRATE_BUILD_INPUTS)
    if changed:
        stale = " and ".join(str(_wheel_dir(crates[side])) for side in ("old", "new"))
        raise AbError(
            f"crates {crates['old'][:7]} and {crates['new'][:7]} differ in {', '.join(changed.splitlines())} "
            f"but built the same extension ({shas['old'][:12]}): a stale build; remove {stale} and run again"
        )


def ensure_wheel(crate: str) -> Path:
    """The cached release wheel of crate commit `crate`, built once from a `git archive` of the
    crate repository (fetched first if the commit is missing) with the toolchain it pins."""
    wheel_dir = _wheel_dir(crate)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if wheels:
        return wheels[-1]
    BUILT.append(crate)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["git", "-C", str(CRATE_REPO), "cat-file", "-e", f"{crate}^{{commit}}"], check=False).returncode:
        _git(CRATE_REPO, "fetch", "--quiet", "origin")
    source = wheel_dir / "src"
    extract_crate(crate, source)
    env = _pass_env(source)
    env.pop("PYTHONPATH")
    env["CARGO_TARGET_DIR"] = str(_wheels_root() / "target")  # shared, so later builds are incremental
    log = wheel_dir / "build.log"
    maturin = Path(sys.executable).parent / "maturin"
    with open(log, "w") as out:
        command = [str(maturin), "build", "--release", "--interpreter", sys.executable, "--out", str(wheel_dir)]
        code = subprocess.run(
            command, check=False, cwd=source, env=env, stdout=out, stderr=subprocess.STDOUT
        ).returncode
    shutil.rmtree(source, ignore_errors=True)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if code or not wheels:
        raise AbError(f"crate {crate[:7]}: maturin build failed (log: {log})", _tail(log))
    return wheels[-1]


def ensure_site(crate: str) -> tuple[Path, str]:
    """(site directory holding crate's extension, the extension's sha256): the cached wheel
    installed with `pip --target`, never into the venv."""
    site = _wheel_dir(crate) / "site"
    if not site.is_dir():
        wheel = ensure_wheel(crate)
        tmp = site.with_name("site.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--no-index",
            "--target",
            str(tmp),
            str(wheel),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode:
            raise AbError(f"crate {crate[:7]}: pip install --target failed", result.stderr.splitlines())
        tmp.rename(site)
    extensions = sorted((site / "indrajala_math_rust").glob("*.so"))
    if not extensions:
        raise AbError(f"crate {crate[:7]}: no extension module in {site}")
    return site, hashlib.sha256(extensions[0].read_bytes()).hexdigest()


def resolve_crate(ref: str) -> str:
    """A crate commit's full sha from the crate repository, fetching once if it isn't there."""
    for attempt in range(2):
        result = subprocess.run(
            ["git", "-C", str(CRATE_REPO), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        if attempt == 0:
            _git(CRATE_REPO, "fetch", "--quiet", "origin")
    raise AbError(f"crate commit {ref!r} not found in {CRATE_REPO}")


# ---- passes


def _pass_env(tree: Path, site: str | None = None) -> dict[str, str]:
    """The environment of a process on one side: its tree, then its crate's site directory when the
    crate is switched, on PYTHONPATH (ahead of the venv's own extension)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree) + (f"{os.pathsep}{site}" if site else "")
    env["PATH"] = f"{CARGO_BIN}{os.pathsep}{env.get('PATH', '')}"
    return env


def check_provenance(run_dir: Path, tree: Path, env: dict[str, str], extension_sha: str | None) -> dict[str, Any]:
    """Where the pass's environment imports from; raises unless the trainer (indrajala_ml.training.train,
    or indrajala_ml.train before the source layout moved it) is tree's and the crate extension's hash
    is extension_sha (when given)."""
    result = subprocess.run(
        [sys.executable, "-c", PROVENANCE_PROBE, str(tree)],
        check=False,
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AbError(f"provenance: the trainer doesn't import from {tree}", result.stderr.splitlines())
    found: dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    if not Path(found["train"]).resolve().is_relative_to(tree.resolve()):
        raise AbError(f"provenance: the trainer imported from {found['train']}, not from {tree}")
    if extension_sha is not None and found["sha256"] != extension_sha:
        raise AbError(
            f"provenance: crate extension {found['extension']} has sha256 {str(found['sha256'])[:12]}, "
            f"not the side's {extension_sha[:12]}"
        )
    return found


def _run_step(
    manifest: dict[str, Any], run_dir: Path, side: str, stem: str, args: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Provenance, then the benchmark once on side, writing stem.{json,stdout,stderr}."""
    adapter = ADAPTERS[manifest["bench"]]
    tree = Path(manifest[side]["tree"])
    site = manifest[side].get("site")
    env = _pass_env(tree, site)
    if site:  # a switched crate: the side's own build, from its site directory
        provenance = check_provenance(run_dir, tree, env, manifest[side]["extension_sha"])
    else:  # both sides on the venv's extension, which must not change during the run
        extension = manifest.get("extension")
        provenance = check_provenance(run_dir, tree, env, extension["sha256"] if extension else None)
        if extension is None:
            manifest["extension"] = {"file": provenance["extension"], "sha256": provenance["sha256"]}
    script_tree = Path(manifest[manifest["script_from"]]["tree"])
    command = [sys.executable, *adapter.command(script_tree, args, run_dir / f"{stem}.json")]
    with open(run_dir / f"{stem}.stdout", "w") as out, open(run_dir / f"{stem}.stderr", "w") as err:
        code = subprocess.run(command, check=False, cwd=run_dir, env=env, stdout=out, stderr=err).returncode
    if code:
        raise AbError(f"{stem} ({side}) exited {code}: {' '.join(command)}", _tail(run_dir / f"{stem}.stderr"))
    return provenance, command


def _run_passes(run_dir: Path, manifest: dict[str, Any], sides: list[str]) -> None:
    first = max((p["number"] for p in manifest["passes"]), default=0) + 1
    for number, side in enumerate(sides, first):
        stem = f"pass-{number:02d}-{side}"
        record: dict[str, Any] = {
            "number": number,
            "side": side,
            "stem": stem,
            "status": "running",
            "start": _now(),
            "load": list(os.getloadavg()),
            "busy": busy_processes(),
        }
        manifest["passes"].append(record)
        _write_json(run_dir / "manifest.json", manifest)
        _append_progress(run_dir, {"event": "start", **record})
        try:
            record["provenance"], record["command"] = _run_step(manifest, run_dir, side, stem, manifest["args"])
            record["status"] = "ok"
        except AbError:
            record["status"] = "failed"
            raise
        finally:
            record["end"] = _now()
            _write_json(run_dir / "manifest.json", manifest)
            _append_progress(run_dir, {"event": "end", "number": number, "status": record["status"], "end": _now()})


def _preflight(run_dir: Path, manifest: dict[str, Any], skip_profile: bool, allow_change: bool) -> dict[str, Any]:
    """The machine check, load and thread settings, and the data symlink; returns the record."""
    record: dict[str, Any] = {
        "at": _now(),
        "load": list(os.getloadavg()),
        "busy": busy_processes(),
        "threads": {var: os.environ.get(var) for var in THREAD_VARS},
    }
    data = Path(manifest["repo"]) / "data"
    if data.is_dir() and not (run_dir / "data").exists():
        (run_dir / "data").symlink_to(data)
    if skip_profile:
        record["profile"] = "skipped"
        return record
    # from the checkout, not a worktree: worktrees have no rust/ submodule, and the crate both sides
    # import is the venv's, built from the checkout's rust/
    tree = Path(manifest["repo"])
    result = subprocess.run(
        [sys.executable, str(tree / "scripts/machine_profile.py"), "compare", str(tree / PROFILE_REFERENCE)],
        check=False,
        cwd=run_dir,
        env=_pass_env(tree),
        capture_output=True,
        text=True,
    )
    lines = (result.stdout + result.stderr).splitlines()
    if result.returncode == 0:
        record["profile"] = "identity matches"
    elif allow_change:
        record["profile"] = "identity changed (allowed)"
        record["profile_output"] = [line for line in lines if "->" in line][:STDERR_TAIL]
    else:
        raise AbError("machine profile: identity changed (--allow-profile-change to run anyway)", lines)
    return record


def _eta(manifest: dict[str, Any], passes: int) -> str:
    """Finish time from the mean pass length of earlier runs with the same benchmark and arguments."""
    seconds: list[float] = []
    for run_dir in _run_dirs():
        other = _manifest(run_dir)
        if other["bench"] == manifest["bench"] and other["args"] == manifest["args"]:
            seconds += [_pass_seconds(p) for p in other["passes"] if p["status"] == "ok"]
    if not seconds:
        return "no ETA (first run of these settings)"
    finish = _local_now() + datetime.timedelta(seconds=passes * statistics.mean(seconds))
    return f"ETA {finish:%H:%M}"


def _pass_seconds(record: dict[str, Any]) -> float:
    start = datetime.datetime.fromisoformat(record["start"])
    return (datetime.datetime.fromisoformat(record["end"]) - start).total_seconds()


def _finish(run_dir: Path, manifest: dict[str, Any], started: float, passes: int) -> None:
    manifest["state"] = "done"
    _write_json(run_dir / "manifest.json", manifest)
    minutes = (time.monotonic() - started) / 60
    print(f"done {run_dir.name}: {passes} passes, {minutes:.0f} min; next: ab.py report --brief")


def _guarded(run_dir: Path, manifest: dict[str, Any], work: Any) -> None:
    """Run work(), marking the run failed (and re-raising) if it raises or is interrupted."""
    try:
        work()
    except AbError, KeyboardInterrupt:
        manifest["state"] = "failed"
        _write_json(run_dir / "manifest.json", manifest)
        raise


# ---- commands


def cmd_run(args: argparse.Namespace, extra: list[str]) -> None:
    repo = Path(args.repo).resolve()
    adapter = ADAPTERS[args.bench]
    sides = parse_order(args.order)
    commits = {
        side: _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
        for side, ref in (("old", args.old), ("new", args.new))
    }
    if args.new == "HEAD" and _git(repo, "status", "--porcelain", "--untracked-files=no", "--", *DIRTY_PATHS):
        raise AbError(f"uncommitted changes under {', '.join(DIRTY_PATHS)}: commit them first (--new is HEAD)")
    crates: dict[str, str | None] = {}
    for side, commit in commits.items():
        ref: str | None = getattr(args, f"{side}_crate")
        crates[side] = resolve_crate(ref) if ref else _crate_commit(repo, commit)
    bench_args = extra or list(adapter.default_args)
    name = args.name or f"{_git(repo, 'rev-parse', '--abbrev-ref', 'HEAD')}-{adapter.name}"
    run_dir = _new_run_dir(name)
    manifest: dict[str, Any] = {
        "version": 1,
        "name": run_dir.name,
        "created": _now(),
        "state": "running",
        "pid": os.getpid(),
        "repo": str(repo),
        "bench": adapter.name,
        "args": bench_args,
        "smoke_args": adapter.smoke_args(bench_args),
        "script_from": args.script_from,
        "control_backend": args.control_backend,
        "order": _order_text(sides),
        "python": sys.executable,
        "cwd": str(run_dir),
        "extension": None,
        "passes": [],
        "extends": [],
    }
    for side in ("old", "new"):
        manifest[side] = {"ref": getattr(args, side), "commit": commits[side], "crate": crates[side]}
    _write_json(run_dir / "manifest.json", manifest)
    started = time.monotonic()

    def work() -> None:
        for side in ("old", "new"):
            manifest[side]["tree"] = str(ensure_worktree(repo, commits[side]))
        if crates["old"] != crates["new"]:  # switch the crate per pass; the venv is never touched
            for side in ("old", "new"):
                crate = crates[side]
                if crate is None:
                    raise AbError(f"the {side} side has no crate commit (no rust/ submodule; give --{side}-crate)")
                site, sha = ensure_site(crate)
                manifest[side]["site"], manifest[side]["extension_sha"] = str(site), sha
            check_crate_builds(
                {side: manifest[side]["crate"] for side in ("old", "new")},
                {side: manifest[side]["extension_sha"] for side in ("old", "new")},
            )
        manifest["preflight"] = _preflight(run_dir, manifest, args.skip_profile, args.allow_profile_change)
        manifest["preflight"]["after_builds"] = list(BUILT)
        _write_json(run_dir / "manifest.json", manifest)
        print(
            f"started {run_dir.name}: old {commits['old'][:7]} new {commits['new'][:7]}, "
            f"{len(sides)} passes ({manifest['order']}), {_eta(manifest, len(sides))}",
            flush=True,
        )
        for side in ("old", "new"):
            _run_step(manifest, run_dir, side, f"smoke-{side}", manifest["smoke_args"])
            smoke = run_dir / f"smoke-{side}"
            if not adapter.rows(smoke.with_suffix(".json"), smoke.with_suffix(".stdout"), None):
                raise AbError(
                    f"smoke ({side}): the benchmark ran but gave no rows", _tail(smoke.with_suffix(".stderr"))
                )
        _run_passes(run_dir, manifest, sides)

    _guarded(run_dir, manifest, work)
    _finish(run_dir, manifest, started, len(sides))


def cmd_extend(args: argparse.Namespace) -> None:
    run_dir = find_run(args.run)
    manifest = _manifest(run_dir)
    if manifest["state"] == "running" and _pid_alive(manifest["pid"]):
        raise AbError(f"{run_dir.name} is still running (pid {manifest['pid']})")
    sides = parse_order(args.order)
    first = max((p["number"] for p in manifest["passes"]), default=0) + 1
    shifted = [f"pass {s.number} ({s.side}, {s.direction})" for s in report_data(run_dir).shifted]
    note = args.note or (f"balances shifted {', '.join(shifted)}" if shifted else "")
    manifest["extends"].append({"order": _order_text(sides), "first_pass": first, "note": note})
    manifest["state"] = "running"
    manifest["pid"] = os.getpid()
    started = time.monotonic()

    def work() -> None:
        manifest["extends"][-1]["preflight"] = _preflight(
            run_dir, manifest, args.skip_profile, args.allow_profile_change
        )
        _write_json(run_dir / "manifest.json", manifest)
        print(
            f"extending {run_dir.name}: passes {first}-{first + len(sides) - 1} ({_order_text(sides)}), {_eta(manifest, len(sides))}",
            flush=True,
        )
        _run_passes(run_dir, manifest, sides)

    _guarded(run_dir, manifest, work)
    _finish(run_dir, manifest, started, len(sides))


def cmd_status(args: argparse.Namespace) -> None:
    run_dir = find_run(args.run)
    manifest = _manifest(run_dir)
    done = [p for p in manifest["passes"] if p["status"] == "ok"]
    state = manifest["state"]
    if state == "running" and not _pid_alive(manifest["pid"]):
        state = "stopped (its process is gone)"
    running = [p for p in manifest["passes"] if p["status"] == "running"]
    line = f"{run_dir.name}: {state}, {len(done)} passes done"
    if manifest["state"] == "running" and running:
        current = running[-1]
        planned = len(manifest["order"]) + sum(len(e["order"]) for e in manifest["extends"])
        line += f", pass {current['number']} ({current['side']}) running since {_clock(current['start'])}"
        if done:
            left = (planned - len(done)) * statistics.mean(_pass_seconds(p) for p in done)
            elapsed = (_local_now() - datetime.datetime.fromisoformat(current["start"])).total_seconds()
            finish = _local_now() + datetime.timedelta(seconds=max(left - elapsed, 0))
            line += f", ETA {finish:%H:%M}"
    print(line)


def cmd_report(args: argparse.Namespace) -> None:
    run_dir = find_run(args.run)
    data = report_data(run_dir)
    if args.md:
        Path(args.md).write_text(markdown_report(data))
        if not args.brief:
            print(f"wrote {args.md}")
    if args.brief or not args.md:
        print("\n".join(brief_report(data)))


def cmd_clean(args: argparse.Namespace) -> None:
    if not (args.worktrees or args.wheels):
        raise AbError("clean needs --worktrees, --wheels or both")
    repo = Path(args.repo).resolve()
    cutoff = _local_now() - datetime.timedelta(days=CLEAN_DAYS)
    referenced: set[Path] = set()
    crates: set[str] = set()
    for run_dir in _run_dirs():
        manifest = _manifest(run_dir)
        recent = datetime.datetime.fromisoformat(manifest["created"]) >= cutoff
        alive = manifest["state"] == "running" and _pid_alive(manifest["pid"])
        if recent or alive:
            referenced |= {Path(manifest[side]["tree"]) for side in ("old", "new") if "tree" in manifest[side]}
            crates |= {manifest[side]["crate"] for side in ("old", "new") if manifest[side].get("site")}
    if args.wheels:
        _clean_wheels(crates)
    if args.worktrees:
        _clean_worktrees(repo, referenced)


def _clean_wheels(crates: set[str]) -> None:
    """Remove the cached wheels and site directories of crate commits no recent run uses (the
    shared cargo target directory stays: it only makes the next build incremental)."""
    removed: list[str] = []
    kept = 0
    for wheel_dir in sorted(_wheels_root().iterdir()) if _wheels_root().is_dir() else []:
        if wheel_dir.name == "target":
            continue
        if wheel_dir.name in crates:
            kept += 1
            continue
        shutil.rmtree(wheel_dir)
        removed.append(wheel_dir.name[:7])
    print(f"removed {len(removed)} wheels ({', '.join(removed) or 'none'}), kept {kept}")


def _clean_worktrees(repo: Path, referenced: set[Path]) -> None:
    removed: list[str] = []
    kept = 0
    for tree in sorted(WORKTREES_ROOT.iterdir()) if WORKTREES_ROOT.is_dir() else []:
        if tree in referenced:
            kept += 1
            continue
        _git(repo, "worktree", "remove", "--force", str(tree))
        removed.append(tree.name)
    _git(repo, "worktree", "prune")
    print(f"removed {len(removed)} worktrees ({', '.join(removed) or 'none'}), kept {kept}")


# ---- the report


@dataclass
class RowStats:
    metric: str
    case: str
    unit: str
    control: bool
    old: list[list[float]]  # per complete old pass, its runs
    new: list[list[float]]

    @property
    def old_runs(self) -> list[float]:
        return [v for runs in self.old for v in runs]

    @property
    def new_runs(self) -> list[float]:
        return [v for runs in self.new for v in runs]

    @property
    def old_median(self) -> float:
        return statistics.median(self.old_runs)

    @property
    def new_median(self) -> float:
        return statistics.median(self.new_runs)

    @property
    def delta(self) -> float:
        return self.new_median / self.old_median - 1 if self.old_median else 0.0

    @property
    def old_pass_medians(self) -> list[float]:
        return [statistics.median(runs) for runs in self.old]

    @property
    def new_pass_medians(self) -> list[float]:
        return [statistics.median(runs) for runs in self.new]

    @property
    def separated(self) -> bool:
        """Every per-pass median of one side beyond every one of the other, with 2+ passes a side."""
        old, new = self.old_pass_medians, self.new_pass_medians
        return len(old) >= 2 and len(new) >= 2 and (max(old) < min(new) or max(new) < min(old))

    @property
    def consistent(self) -> bool:
        """Separated, and the gap between the sides is wider than each side's spread of pass medians."""
        if not self.separated:
            return False
        old, new = self.old_pass_medians, self.new_pass_medians
        gap = max(min(new) - max(old), min(old) - max(new))
        return gap > max(max(old) - min(old), max(new) - min(new))


@dataclass(frozen=True)
class ShiftedPass:
    number: int
    side: str
    ratio: float  # the median over rows of pass median / the side's pooled median

    @property
    def direction(self) -> str:
        return f"{'fast' if self.ratio < 1 else 'slow'} {(self.ratio - 1) * 100:+.1f}%"


@dataclass
class ReportData:
    manifest: dict[str, Any]
    rows: list[RowStats]
    passes: dict[str, list[int]]  # complete pass numbers per side
    shifted: list[ShiftedPass]
    one_sided: int  # (metric, case) pairs only one side measured


def report_data(run_dir: Path) -> ReportData:
    manifest = _manifest(run_dir)
    adapter = ADAPTERS[manifest["bench"]]
    passes: dict[str, list[int]] = {"old": [], "new": []}
    per_pass: dict[tuple[str, str], dict[int, list[float]]] = {}
    meta: dict[tuple[str, str], tuple[str, bool]] = {}
    for record in manifest["passes"]:
        stem = run_dir / record["stem"]
        if record["status"] != "ok":
            continue
        passes[record["side"]].append(record["number"])
        for row in adapter.rows(stem.with_suffix(".json"), stem.with_suffix(".stdout"), manifest["control_backend"]):
            key = (row.metric, row.case)
            per_pass.setdefault(key, {}).setdefault(record["number"], []).append(row.value)
            meta[key] = (row.unit, row.control)
    rows: list[RowStats] = []
    one_sided = 0
    for key, by_pass in per_pass.items():
        old = [by_pass[n] for n in passes["old"] if n in by_pass]
        new = [by_pass[n] for n in passes["new"] if n in by_pass]
        if not old or not new:
            one_sided += 1
            continue
        rows.append(RowStats(key[0], key[1], meta[key][0], meta[key][1], old, new))
    return ReportData(manifest, rows, passes, _shifted_passes(rows, passes), one_sided)


def _shifted_passes(rows: list[RowStats], passes: dict[str, list[int]]) -> list[ShiftedPass]:
    shifted: list[ShiftedPass] = []
    for side, numbers in passes.items():
        for index, number in enumerate(numbers):
            ratios: list[float] = []
            control_ratios: list[float] = []
            for row in rows:
                runs = row.old if side == "old" else row.new
                pooled = row.old_median if side == "old" else row.new_median
                if index < len(runs) and pooled:
                    ratio = statistics.median(runs[index]) / pooled
                    ratios.append(ratio)
                    if row.control:
                        control_ratios.append(ratio)
            if not ratios:
                continue
            ratio = statistics.median(ratios)
            control = statistics.median(control_ratios) if control_ratios else ratio
            if abs(ratio - 1) >= SHIFT and abs(control - 1) >= SHIFT and (ratio < 1) == (control < 1):
                shifted.append(ShiftedPass(number, side, ratio))
    return sorted(shifted, key=lambda s: s.number)


def balancing_order(shifted: list[ShiftedPass]) -> str:
    """Pairs to add so each side has as many fast and as many slow shifted passes: each pair
    starts with the side that has fewer, e.g. NO for one fast old pass. Empty when balanced."""
    order = ""
    for fast in (True, False):
        old = sum(1 for s in shifted if s.side == "old" and (s.ratio < 1) == fast)
        new = sum(1 for s in shifted if s.side == "new" and (s.ratio < 1) == fast)
        order += ("NO" if old > new else "ON") * abs(old - new)
    return order


def _value(value: float, unit: str) -> str:
    return f"{value:.3f}" if unit == "s" else f"{value:.4g}"


def _delta(delta: float) -> str:
    return f"{delta * 100:+.1f}%"


def _row_label(row: RowStats) -> str:
    return f"{row.metric} / {row.case}"


def _commits_line(manifest: dict[str, Any], data: ReportData) -> str:
    numbers = sorted(data.passes["old"] + data.passes["new"])
    order = "".join("O" if n in data.passes["old"] else "N" for n in numbers)
    command = (
        " ".join([manifest["bench"], *manifest["args"]]) if manifest["bench"] != "cmd" else " ".join(manifest["args"])
    )
    line = (
        f"{manifest['name']}: old {manifest['old']['commit'][:7]} vs new {manifest['new']['commit'][:7]}, "
        f"passes {order} ({len(data.passes['old'])} old, {len(data.passes['new'])} new), {command}"
    )
    if manifest["old"].get("site"):
        line += "; crate " + " vs ".join(
            f"{side} {manifest[side]['crate'][:7]} (.so {manifest[side]['extension_sha'][:12]})"
            for side in ("old", "new")
        )
    return line


def _machine_line(manifest: dict[str, Any]) -> str:
    checks = [manifest.get("preflight", {})] + [e.get("preflight", {}) for e in manifest["extends"]]
    profiles = sorted({c.get("profile", "not recorded") for c in checks})
    # pre-flight only: before a later pass the 1-minute load still counts the previous pass's benchmark
    loads = [c["load"][0] for c in checks if "load" in c]
    busy = sorted(
        {b.split("(")[0] for c in checks for b in c.get("busy", [])}
        | {b.split("(")[0] for p in manifest["passes"] for b in p.get("busy", [])}
    )
    line = f"profile: {', '.join(profiles)}"
    if loads:
        line += f"; max 1-min load {max(loads):.2f}" + (" (HIGH)" if max(loads) > HIGH_LOAD else "")
        if any(c.get("after_builds") for c in checks):
            line += " (measured just after this run's crate builds)"
    return line + f"; busy processes: {', '.join(busy[:4]) or 'none'}"


def brief_report(data: ReportData) -> list[str]:
    """At most BRIEF_LINES lines: header, machine, control and shift findings, consistent rows, a summary."""
    manifest = data.manifest
    lines = [_commits_line(manifest, data), _machine_line(manifest)]
    if min(len(data.passes["old"]), len(data.passes["new"])) < 2:
        lines.append("too few complete passes for verdicts: they need 2+ a side (ab.py extend)")
    controls = [row for row in data.rows if row.control]
    moved = [row for row in controls if row.consistent]
    if moved:
        worst = max(moved, key=lambda row: abs(row.delta))
        lines.append(
            f"CONTROL MOVED: {len(moved)} of {len(controls)} control rows consistent, e.g. {_row_label(worst)} "
            f"{_delta(worst.delta)}: changes about that size can't be resolved here (next: epoch_op_profile.py)"
        )
    elif controls:
        lines.append(f"controls: {len(controls)} rows, all within noise")
    else:
        lines.append(ADAPTERS[manifest["bench"]].no_control)
    if data.shifted:
        balance = balancing_order(data.shifted)
        described = ", ".join(f"pass {s.number} ({s.side}, {s.direction})" for s in data.shifted)
        lines.append(
            f"shifted: {described}; " + (f"unbalanced, next: ab.py extend --order {balance}" if balance else "balanced")
        )
    consistent = sorted((row for row in data.rows if row.consistent and not row.control), key=lambda r: -abs(r.delta))
    room = BRIEF_LINES - len(lines) - 1
    shown = consistent if len(consistent) <= room else consistent[: room - 1]
    for row in shown:
        lines.append(
            f"consistent: {_row_label(row)} {_delta(row.delta)} "
            f"({_value(row.old_median, row.unit)} -> {_value(row.new_median, row.unit)} {row.unit})"
        )
    if len(shown) < len(consistent):
        lines.append(f"... {len(consistent) - len(shown)} more consistent rows (report --md)")
    noise = [row for row in data.rows if not row.consistent and not row.control]
    summary = f"{len(noise)} rows within noise"
    if noise:
        separated = sum(1 for row in noise if row.separated)
        summary += f" ({separated} separated inside their spread)" if separated else ""
        summary += f", max |Δ| {max(abs(row.delta) for row in noise) * 100:.1f}%"
    if data.one_sided:
        summary += f"; {data.one_sided} rows measured on one side only"
    lines.append(summary)
    return lines


def _table(rows: list[RowStats], data: ReportData) -> list[str]:
    old_numbers = ", ".join(str(n) for n in data.passes["old"])
    new_numbers = ", ".join(str(n) for n in data.passes["new"])
    unit = rows[0].unit
    lines = [
        (
            f"| case | old median (min-max) {unit} | new median (min-max) {unit} | Δ median "
            f"| per-pass medians old ({old_numbers}) / new ({new_numbers}) | verdict |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for row in rows:

        def pooled(runs: list[float], median: float) -> str:
            return f"{_value(median, unit)} ({_value(min(runs), unit)}-{_value(max(runs), unit)})"

        per_pass = (
            ", ".join(_value(v, unit) for v in row.old_pass_medians)
            + " / "
            + ", ".join(_value(v, unit) for v in row.new_pass_medians)
        )
        verdict = "consistent" if row.consistent else ("separated, inside spread" if row.separated else "within noise")
        if row.control:
            verdict += " (control)"
        lines.append(
            f"| {row.case} | {pooled(row.old_runs, row.old_median)} | {pooled(row.new_runs, row.new_median)} "
            f"| {_delta(row.delta)} | {per_pass} | {verdict} |"
        )
    return lines


def protocol_paragraph(data: ReportData) -> str:
    manifest = data.manifest
    adapter = ADAPTERS[manifest["bench"]]
    script_tree = Path(manifest.get(manifest["script_from"], {}).get("tree", "<tree>"))
    command = " ".join(["python", *adapter.command(script_tree, manifest["args"], Path("<pass output>"))])
    command = command.replace(str(script_tree) + "/", "")
    extension: dict[str, Any] = manifest.get("extension") or {}
    text = (
        f"Old `{manifest['old']['commit'][:7]}` against new `{manifest['new']['commit'][:7]}`, run with "
        f"`scripts/ab.py`: passes in the order {manifest['order']}"
    )
    for extend in manifest["extends"]:
        last = extend["first_pass"] + len(extend["order"]) - 1
        text += f", then passes {extend['first_pass']}-{last} ({extend['order']})"
        text += f", which {extend['note']}" if extend["note"] else ""
    runs = [len(row.old_runs) for row in data.rows[:1]] + [len(row.new_runs) for row in data.rows[:1]]
    text += (
        f". Each pass ran `{command}` (the script from the {manifest['script_from']} tree) in its own process "
        f"tree, from a neutral working directory with only its side's tree on `PYTHONPATH`. Before each pass a "
        f"probe checked that the trainer module imported from that tree"
    )
    if manifest["old"].get("site"):
        builds = " and ".join(
            f"{side} crate `{manifest[side]['crate'][:7]}` (extension sha256 `{manifest[side]['extension_sha'][:12]}`)"
            for side in ("old", "new")
        )
        text += (
            f" and that the crate extension was that side's release build: {builds}, each a cached wheel "
            "installed with `pip --target` into a directory on `PYTHONPATH` after the tree, so the venv was never touched"
        )
    elif extension.get("sha256"):
        text += f" and that the crate extension was the venv's (sha256 `{extension['sha256'][:12]}`) on both sides"
    text += f". Machine check: {_machine_line(manifest)}."
    if runs:
        pooled = f"{runs[0]} runs per side" if runs[0] == runs[-1] else f"{runs[0]} old and {runs[-1]} new runs"
        text += f" Each row pools {pooled}."
    return text


def markdown_report(data: ReportData) -> str:
    lines = [protocol_paragraph(data), ""]
    lines += [f"- {line}" for line in brief_report(data)[2:]]
    metrics: list[str] = []
    for row in data.rows:
        if row.metric not in metrics:
            metrics.append(row.metric)
    for metric in metrics:
        lines += ["", f"### {metric}", ""]
        lines += _table([row for row in data.rows if row.metric == metric], data)
    return "\n".join(lines) + "\n"


# ---- entry point


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=str(REPO), help="the repository whose commits are compared")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run an A/B")
    run.add_argument("--bench", required=True, choices=sorted(ADAPTERS))
    run.add_argument("--old", default="main")
    run.add_argument("--new", default="HEAD")
    run.add_argument("--order", default="ONNONO")
    run.add_argument("--name", help="default <branch>-<bench>")
    run.add_argument("--control-backend", choices=["numpy", "rust"], help="also read this backend's rows as controls")
    run.add_argument("--script-from", choices=["old", "new"], default="new")
    run.add_argument("--old-crate", help="crate commit for the old side (default: its tree's rust/ submodule)")
    run.add_argument("--new-crate", help="crate commit for the new side (default: its tree's rust/ submodule)")
    for command in (run, extend := commands.add_parser("extend", help="add passes to a finished run")):
        command.add_argument("--allow-profile-change", action="store_true")
        command.add_argument("--skip-profile", action="store_true", help="skip the machine check (tests, toy probes)")
    extend.add_argument("run", nargs="?")
    extend.add_argument("--order", required=True)
    extend.add_argument("--note", help="why the passes were added (default: the shifted passes they balance)")

    status = commands.add_parser("status", help="one line: the run's progress")
    status.add_argument("run", nargs="?")
    report = commands.add_parser("report", help="the pooled table and verdicts")
    report.add_argument("run", nargs="?")
    report.add_argument("--brief", action="store_true", help=f"at most {BRIEF_LINES} lines (the default output)")
    report.add_argument("--md", help="write the full tables and the protocol paragraph here")
    clean = commands.add_parser("clean", help="remove worktrees no recent run refers to")
    clean.add_argument("--worktrees", action="store_true")
    clean.add_argument("--wheels", action="store_true", help="cached crate wheels and their site directories")

    args = parser.parse_args(argv)
    if extra and args.command != "run":
        parser.error("arguments after -- are for run only")
    try:
        if args.command == "run":
            cmd_run(args, extra)
        elif args.command == "extend":
            cmd_extend(args)
        elif args.command == "status":
            cmd_status(args)
        elif args.command == "report":
            cmd_report(args)
        else:
            cmd_clean(args)
    except AbError as error:
        print(f"ab.py {args.command}: {error}", file=sys.stderr)
        for line in error.tail:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
