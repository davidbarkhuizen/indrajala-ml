"""
The batch runner (docs/batch-runner-workplan.md, stage 3): a controller that owns a batch's attempts
and its ledger, and a worker daemon on each machine that runs them, over newline-delimited JSON on a
TCP socket (D4). The standard library only, so the worker costs almost nothing at rest.

A batch is a directory holding `ledger.jsonl`: its first line the batch's registration (`batch`:
the mode, the cell, the attempts with their tasks, seeds and cutoffs), then one line per event
(`assigned`, `started`, `ended`, `lost`, `done`). The controller rebuilds its state from it on start
(D5), so it resumes after its own restart. Each attempt names a registered cell (CELLS), never a
command line, so a worker runs nothing else.

Messages (each a JSON object on one line, with `type` and the protocol's version `v`):
worker to controller `hello` (host, slots, the attempts it is running), `ready` (free slots),
`started` (an attempt's pid), `progress` (its last step), `ended` (its summary); controller to
worker `attempt` (an attempt to run), `drain` (no more work).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VERSION = 1
LEDGER = "ledger.jsonl"
POLL = 2.0  # seconds between a worker's looks at its attempts

# The continuous crystallisation's cell (D54's, with the margin stop under its deadline: D59, D60),
# as probe-d64.sh runs it; "-reheat" adds D63 (a).
CONTINUOUS_CELL = [
    "--factors",
    "2",
    "--codes-ratio",
    "32/11",
    "--reads",
    "output",
    "--window-width",
    "32",
    "--budget-epochs",
    "1600",
    "--budget-minutes",
    "690",
    "--anneal",
    "--anneal-tail",
    "40",
    "--anneal-never-commit",
    "--warm-up",
    "--melt-hot",
    "2",
    "--melt-kappa",
    "1e-2",
    "--melt-hold",
    "20",
    "--melt-halving",
    "30",
    "--melt-shape",
    "exponential",
    "--crucible",
    "--warm-up-screen",
    "1.01",
    "--continuous",
    "--workers",
    "1",
    "--margin-wait",
    "200",
]
CELLS: dict[str, tuple[str, ...]] = {
    "continuous": tuple(CONTINUOUS_CELL),
    "continuous-reheat": (*CONTINUOUS_CELL, "--reheat"),
}
TASKS: dict[str, tuple[str, int]] = {  # a backdrop's short name: the task's and its state bound
    "s3": ("word-problem-s3", 6),
    "s4": ("word-problem-s4", 24),
    "mtf4": ("move-to-front-4", 24),
    "rq12": ("random-q12-0", 12),
    "rq16": ("random-q16-0", 16),
}


@dataclass(frozen=True)
class Attempt:
    """One attempt: a backdrop's seed, cut at `cutoff` steps (None: run to its budget)."""

    id: str
    backdrop: str
    seed: int
    cutoff: int | None = None


@dataclass(frozen=True)
class Batch:
    """
    A batch's registration (D3, D8): `measure` runs every attempt; `find` runs each backdrop's
    attempts in order until `k` of them succeed (the rest never start).
    """

    name: str
    mode: str
    cell: str
    attempts: tuple[Attempt, ...]
    k: int | None = None

    def __post_init__(self) -> None:
        assert self.mode in ("measure", "find"), f"a mode: measure or find; got {self.mode}"
        assert self.mode != "find" or (self.k is not None and self.k >= 1), "find needs k >= 1"
        assert self.cell in CELLS, f"a registered cell: one of {sorted(CELLS)}"
        assert all(a.backdrop in TASKS for a in self.attempts), f"backdrops: {sorted(TASKS)}"
        assert len({a.id for a in self.attempts}) == len(self.attempts), "attempts' ids unique"

    def to_json(self) -> dict[str, Any]:
        return {"type": "batch", "v": VERSION, **asdict(self)}

    @staticmethod
    def from_json(row: dict[str, Any]) -> Batch:
        attempts = tuple(Attempt(**a) for a in row["attempts"])
        return Batch(row["name"], row["mode"], row["cell"], attempts, row.get("k"))


def attempts(spec: str, cutoffs: dict[str, int] | None = None) -> tuple[Attempt, ...]:
    """Attempts from `s3:40-47,mtf4:32-35`, each backdrop cut at its `cutoffs` entry if given."""
    made: list[Attempt] = []
    for part in spec.split(","):
        backdrop, seeds = part.split(":")
        first, _, last = seeds.partition("-")
        for seed in range(int(first), int(last or first) + 1):
            made.append(Attempt(f"{backdrop}-{seed}", backdrop, seed, (cutoffs or {}).get(backdrop)))
    return tuple(made)


def register(directory: Path, batch: Batch) -> None:
    """A new batch's directory and its ledger's first line (refused if it exists: never two registrations)."""
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / LEDGER
    assert not ledger.exists(), f"{ledger} exists: a batch is registered once"
    ledger.write_text(json.dumps(batch.to_json()) + "\n")


@dataclass
class State:
    """A batch's state as its ledger has it."""

    batch: Batch
    assigned: dict[str, str] = field(default_factory=dict[str, str])  # attempt -> host, not yet ended
    ended: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])  # attempt -> summary
    done: bool = False

    @staticmethod
    def read(directory: Path) -> State:
        """The state from the ledger (D5): every event in order."""
        lines = (directory / LEDGER).read_text().splitlines()
        state = State(Batch.from_json(json.loads(lines[0])))
        for line in lines[1:]:
            state.apply(json.loads(line))
        return state

    def apply(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "assigned":
            self.assigned[event["id"]] = event["host"]
        elif kind == "ended":
            self.assigned.pop(event["id"], None)
            self.ended[event["id"]] = event["summary"]
        elif kind == "lost":
            self.assigned.pop(event["id"], None)
        elif kind == "done":
            self.done = True

    def succeeded(self, backdrop: str) -> int:
        return sum(
            1
            for a in self.batch.attempts
            if a.backdrop == backdrop and self.ended.get(a.id, {}).get("success") is not None
        )

    def next_attempt(self) -> Attempt | None:
        """The first attempt neither assigned nor ended, its backdrop not yet found (find mode)."""
        for a in self.batch.attempts:
            if a.id in self.assigned or a.id in self.ended:
                continue
            if self.batch.mode == "find" and self.succeeded(a.backdrop) >= (self.batch.k or 0):
                continue
            return a
        return None

    def finished(self) -> bool:
        return not self.assigned and self.next_attempt() is None


def summary(log: Path) -> dict[str, Any]:
    """
    An attempt's outcome from its run's log.jsonl: its last step's end and seconds, and its success
    (D41's: the first step's end certified at both verdicts; None if never).
    """
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    steps = [r for r in rows if "end" in r and "state_share" in r]
    if not steps:
        return {"success": None, "step": None, "end": "error", "seconds": None}
    success = next(
        (
            int(r["step"])
            for r in steps
            if r.get("certified") and r.get("certificates", {}).get("strict", {}).get("certified")
        ),
        None,
    )
    last = steps[-1]
    return {"success": success, "step": int(last["step"]), "end": last["end"], "seconds": float(last["seconds"])}


def last_step(log: Path) -> int | None:
    """The run's last logged step, read from the log's end (cheap on a long log)."""
    if not log.exists():
        return None
    with log.open("rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(f.tell() - 65536, 0))
        tail = f.read().decode(errors="replace").splitlines()[1:]
    for line in reversed(tail):
        with contextlib.suppress(ValueError):
            return int(json.loads(line)["step"])
    return None


async def send(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    writer.write((json.dumps({**message, "v": VERSION}) + "\n").encode())
    await writer.drain()


class Controller:
    """
    A batch's controller (D4): the protocol's server on 127.0.0.1, its queue and its ledger.
    Each attempt is assigned once; a worker's `ready` gets as many as it has free slots.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.state = State.read(directory)
        self.finished = asyncio.Event()
        if self.state.finished():
            self.finished.set()

    def log(self, event: dict[str, Any]) -> None:
        with (self.directory / LEDGER).open("a") as f:
            f.write(json.dumps({**event, "t": round(time.time(), 3)}) + "\n")
        self.state.apply(event)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        host = "?"
        try:
            while line := await reader.readline():
                message = json.loads(line)
                if message.get("v") != VERSION:
                    break
                kind = message["type"]
                if kind == "hello":
                    host = message["host"]
                    running = set(message.get("running", []))
                    for attempt, owner in list(self.state.assigned.items()):  # a host's attempts it no longer has
                        if owner == host and attempt not in running:
                            self.log({"type": "lost", "id": attempt, "host": host})
                elif kind == "ready":
                    for _ in range(int(message["free"])):
                        attempt = self.state.next_attempt()
                        if attempt is None:
                            break
                        self.log({"type": "assigned", "id": attempt.id, "host": host})
                        await send(writer, {"type": "attempt", "cell": self.state.batch.cell, **asdict(attempt)})
                    if self.state.next_attempt() is None:
                        await send(writer, {"type": "drain"})
                elif kind == "started":
                    self.log({"type": "started", "id": message["id"], "host": host, "pid": message["pid"]})
                elif kind == "ended":
                    self.log({"type": "ended", "id": message["id"], "host": host, "summary": message["summary"]})
                if self.state.finished() and not self.state.done:
                    self.log({"type": "done"})
                    self.finished.set()
        finally:
            writer.close()

    async def serve(self, port: int) -> None:
        """Serve on 127.0.0.1:port until the batch is done (every attempt ended or never to start)."""
        server = await asyncio.start_server(self.handle, "127.0.0.1", port)
        async with server:
            await self.finished.wait()


@dataclass
class Running:
    """An attempt a worker runs: its directory (its pid file, its run's log) and its process."""

    attempt: Attempt
    directory: Path
    pid: int
    step: int | None = None


class Worker:
    """
    A machine's worker daemon (D4): connects to the controller, asks for attempts while it has free
    slots, runs each as its own detached session (pid file, output beside it), reports its start,
    progress and end, and re-adopts its attempts from their pid files after its own restart. Its runs
    survive a dropped connection: it reconnects and reports.
    """

    def __init__(
        self,
        root: Path,
        slots: int,
        python: str,
        code: Path,
        script: str = "scripts/modular_family_probe.py",
        host: str | None = None,
        env: dict[str, str] | None = None,
        poll: float = POLL,
    ) -> None:
        self.root, self.slots, self.python, self.code, self.script = root, slots, python, code, script
        self.poll = poll
        self.host = host or socket.gethostname()
        self.env = env or {}
        self.running: dict[str, Running] = {}
        self.draining = False
        root.mkdir(parents=True, exist_ok=True)
        self.adopt()

    def adopt(self) -> None:
        """Its attempts still running from before a restart, from their pid files."""
        for pid_file in self.root.glob("*/pid"):
            record = json.loads(pid_file.read_text())
            if alive(record["pid"]) and not (pid_file.parent / "ended.json").exists():
                attempt = Attempt(**record["attempt"])
                self.running[attempt.id] = Running(attempt, pid_file.parent, record["pid"])

    def command(self, attempt: Attempt, cell: str) -> list[str]:
        task, bound = TASKS[attempt.backdrop]
        cutoff = ["--cutoff-steps", str(attempt.cutoff)] if attempt.cutoff is not None else []
        return [
            self.python,
            self.script,
            "run",
            "--runs",
            f"{task}:{attempt.seed}",
            *CELLS[cell],
            "--state-bound",
            str(bound),
            *cutoff,
            "--out",
            str(self.root / attempt.id),
        ]

    def launch(self, attempt: Attempt, cell: str) -> Running:
        directory = self.root / attempt.id
        directory.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            **self.env,
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "AB_RUNS_ROOT": str(directory / "ab-runs"),  # each run its own machine lock
        }
        with (directory / "run.out").open("ab") as out:  # its own session: it outlives the worker
            child = subprocess.Popen(
                self.command(attempt, cell),
                cwd=self.code,
                env=env,
                stdout=out,
                stderr=out,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        (directory / "pid").write_text(json.dumps({"pid": child.pid, "attempt": asdict(attempt)}))
        running = Running(attempt, directory, child.pid)
        self.running[attempt.id] = running
        return running

    def log_of(self, running: Running) -> Path:
        # one run's log: --runs NAME:SEED writes out/seed<s>/log.jsonl
        return running.directory / f"seed{running.attempt.seed}" / "log.jsonl"

    async def session(self, port: int) -> bool:
        """One connection: hello, then work until drained with nothing running; whether to stop."""
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            await send(writer, {"type": "hello", "host": self.host, "slots": self.slots, "running": list(self.running)})
            asked = 0
            while True:
                free = self.slots - len(self.running)
                if free > 0 and not self.draining and asked == 0:
                    await send(writer, {"type": "ready", "free": free})
                    asked = free
                with contextlib.suppress(TimeoutError):
                    line = await asyncio.wait_for(reader.readline(), timeout=self.poll)
                    if not line:
                        return False  # the controller went away: reconnect
                    message = json.loads(line)
                    if message["type"] == "attempt":
                        fields = {k: message[k] for k in ("id", "backdrop", "seed", "cutoff")}
                        running = self.launch(Attempt(**fields), message["cell"])
                        await send(writer, {"type": "started", "id": running.attempt.id, "pid": running.pid})
                        asked = max(asked - 1, 0)
                    elif message["type"] == "drain":
                        self.draining, asked = True, 0
                await self.look(writer)
                if self.draining and not self.running:
                    return True
                if asked and len(self.running) >= self.slots:
                    asked = 0
        finally:
            writer.close()

    async def look(self, writer: asyncio.StreamWriter) -> None:
        """Each running attempt's progress, and the ended ones' summaries."""
        for attempt_id, running in list(self.running.items()):
            step = last_step(self.log_of(running))
            if step != running.step:
                running.step = step
                await send(writer, {"type": "progress", "id": attempt_id, "step": step})
            if not alive(running.pid):
                result = summary(self.log_of(running))
                (running.directory / "ended.json").write_text(json.dumps(result))
                await send(writer, {"type": "ended", "id": attempt_id, "summary": result})
                del self.running[attempt_id]

    async def serve(self, port: int, retry: float = 5.0) -> None:
        """Connect, work, and reconnect after a dropped connection, until drained."""
        while True:
            with contextlib.suppress(OSError):
                if await self.session(port):
                    return
            await asyncio.sleep(retry)


def alive(pid: int) -> bool:
    """Whether a process lives (a finished child reaped, so it doesn't linger as a zombie)."""
    with contextlib.suppress(ChildProcessError):
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def report(directory: Path) -> list[str]:
    """A batch's reading: each backdrop's attempts, certificates and Wilson interval."""
    state = State.read(directory)
    lines = [f"{state.batch.name}: {state.batch.mode}, cell {state.batch.cell}, {'done' if state.done else 'running'}"]
    for backdrop in sorted({a.backdrop for a in state.batch.attempts}):
        mine = [a for a in state.batch.attempts if a.backdrop == backdrop]
        ended = [state.ended[a.id] for a in mine if a.id in state.ended]
        wins = sum(1 for s in ended if s["success"] is not None)
        low, high = wilson(wins, len(ended))
        lines.append(
            f"  {backdrop}: {len(ended)} of {len(mine)} ended, {wins} certified"
            + (f" ({wins / len(ended):.2f}, {low:.2f}-{high:.2f})" if ended else "")
        )
    return lines


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(centre - half, 0.0), min(centre + half, 1.0)


def runs_of(summaries: Sequence[dict[str, Any]]) -> list[tuple[float | None, float, bool]]:
    """Ended attempts as restart_cost's runs (success step, last step, censored at its budget)."""
    return [
        (None if s["success"] is None else float(s["success"]), float(s["step"]), s["end"] == "spent")
        for s in summaries
        if s["step"] is not None
    ]
