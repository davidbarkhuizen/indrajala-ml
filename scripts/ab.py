"""
Timing A/Bs between two commits, run and reported by the protocol in
docs/optimizations/measurement.md (docs/ab-harness-workplan.md has the design):

    python scripts/ab.py run --bench prepared_dataset_timing [--old main] [--new HEAD] [--order ONNONO]
                             [--name NAME] [--control-backend numpy] [--script-from new]
                             [--allow-profile-change] [-- <benchmark arguments>]
    python scripts/ab.py run --bench cmd [...] -- probe.py [probe arguments]
    python scripts/ab.py status [RUN]
    python scripts/ab.py extend [RUN] --order NO
    python scripts/ab.py report [RUN] [--brief] [--md FILE]
    python scripts/ab.py clean --worktrees

Each side is a commit, checked out once as a detached worktree under ~/code/ab-worktrees/<sha7>.
A run lives in ~/code/ab-runs/<YYYY-MM-DD>-<name>/ (manifest.json, progress.jsonl, and each pass's
raw output and logs), which is also the neutral working directory every pass runs from, with a
`data` symlink to this checkout's data/. A pass runs the benchmark once, in its own process tree,
with only its side's tree on PYTHONPATH; a probe first checks that indrajala_ml.train comes from
that tree and that the crate extension's hash is the run's. `run` does a smoke run of each side
with the benchmark's smallest settings before the passes. RUN defaults to the most recent run.

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
The crate is the venv's on both sides; commits whose rust/ submodules differ are refused.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
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

# printed as JSON by a process in a pass's environment: where indrajala_ml.train and the crate
# extension resolve from, and the extension's hash
PROVENANCE_PROBE = r"""
import hashlib, importlib.util, json, pathlib
import indrajala_ml.train as train
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

    def smoke_args(self, args: list[str]) -> list[str]: ...

    def command(self, script_tree: Path, args: list[str], out: Path) -> list[str]:
        """The arguments after `python`: the script, from script_tree, writing its output to out."""
        ...

    def rows(self, out: Path, stdout: Path, control_backend: str | None) -> list[Row]: ...


class PreparedDatasetTiming:
    name: str = "prepared_dataset_timing"
    default_args: tuple[str, ...] = ("--repeats", "5")

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


ADAPTERS: dict[str, Adapter] = {adapter.name: adapter for adapter in (PreparedDatasetTiming(), CommandProbe())}


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


# ---- passes


def _pass_env(tree: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env["PATH"] = f"{CARGO_BIN}{os.pathsep}{env.get('PATH', '')}"
    return env


def check_provenance(run_dir: Path, tree: Path, env: dict[str, str], extension_sha: str | None) -> dict[str, Any]:
    """Where the pass's environment imports from; raises unless indrajala_ml.train is tree's and the
    crate extension's hash is extension_sha (when given)."""
    result = subprocess.run(
        [sys.executable, "-c", PROVENANCE_PROBE], check=False, cwd=run_dir, env=env, capture_output=True, text=True
    )
    if result.returncode:
        raise AbError(f"provenance: indrajala_ml.train doesn't import from {tree}", result.stderr.splitlines())
    found: dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    if not Path(found["train"]).resolve().is_relative_to(tree.resolve()):
        raise AbError(f"provenance: indrajala_ml.train imported from {found['train']}, not from {tree}")
    if extension_sha is not None and found["sha256"] != extension_sha:
        raise AbError(f"provenance: crate extension {found['extension']} hash changed during the run")
    return found


def _run_step(
    manifest: dict[str, Any], run_dir: Path, side: str, stem: str, args: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Provenance, then the benchmark once on side, writing stem.{json,stdout,stderr}."""
    adapter = ADAPTERS[manifest["bench"]]
    tree = Path(manifest[side]["tree"])
    env = _pass_env(tree)
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
    crates = {side: _crate_commit(repo, commit) for side, commit in commits.items()}
    if crates["old"] != crates["new"]:
        raise AbError("the sides' rust/ submodules differ: crate A/Bs arrive in stage 3 of the workplan")
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
        manifest["preflight"] = _preflight(run_dir, manifest, args.skip_profile, args.allow_profile_change)
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
    if not args.worktrees:
        raise AbError("clean needs --worktrees (--wheels arrives with crate A/Bs, stage 3)")
    repo = Path(args.repo).resolve()
    cutoff = _local_now() - datetime.timedelta(days=CLEAN_DAYS)
    referenced: set[Path] = set()
    for run_dir in _run_dirs():
        manifest = _manifest(run_dir)
        recent = datetime.datetime.fromisoformat(manifest["created"]) >= cutoff
        alive = manifest["state"] == "running" and _pid_alive(manifest["pid"])
        if recent or alive:
            referenced |= {Path(manifest[side]["tree"]) for side in ("old", "new") if "tree" in manifest[side]}
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
    return (
        f"{manifest['name']}: old {manifest['old']['commit'][:7]} vs new {manifest['new']['commit'][:7]}, "
        f"passes {order} ({len(data.passes['old'])} old, {len(data.passes['new'])} new), {command}"
    )


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
        lines.append("controls: none")
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
        f"probe checked that `indrajala_ml.train` imported from that tree"
    )
    if extension.get("sha256"):
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
