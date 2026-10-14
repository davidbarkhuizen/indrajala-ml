"""
One heavy job at a time on a machine (the study-runs workplan's D1 and D2, retired: git show
868838e:docs/study-runs-workplan.md): an A/B's run or extend (scripts/ab.py) or a study's training
stage. Each holds an flock on RUNS_ROOT/run.lock for as long as its process lives, so a killed job
never leaves it held, and writes one JSON line naming itself (what, kind, pid, since); a second is
refused with that line. RUNS_ROOT is ab.py's runs directory, AB_RUNS_ROOT or ~/code/ab-runs, read
when the lock is taken.

The lock's file is opened non-inheritable (PEP 446), so a spawned or forkserver worker doesn't hold
it: the job's own process does, and the lock ends with it. A study's workers don't: killed alone,
its process frees the lock while each worker trains on to the end of the run it holds, so stop a
study by its whole process group.
"""

import contextlib
import datetime
import fcntl
import json
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Literal

LOCK_NAME = "run.lock"


class MachineBusy(Exception):
    pass


def runs_root() -> Path:
    return Path(os.environ.get("AB_RUNS_ROOT", Path.home() / "code/ab-runs"))


def describe(record: str) -> str:
    """A holder's line for a person; an old ab.py's plain-text line as it is."""
    try:
        fields = json.loads(record)
    except ValueError:
        return record.strip()
    return f"{fields['kind']} {fields['what']}, pid {fields['pid']}, since {fields['since']}"


@contextlib.contextmanager
def hold(what: str, kind: Literal["run", "study"], root: Path | None = None) -> Generator[None]:
    """The machine for as long as the block runs; MachineBusy, naming the holder, if it's taken."""
    root = runs_root() if root is None else root
    root.mkdir(parents=True, exist_ok=True)
    with open(root / LOCK_NAME, "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.seek(0)
            raise MachineBusy(f"{root / LOCK_NAME} is held by {describe(lock.read())}") from None
        lock.truncate(0)
        since = datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds")
        lock.write(json.dumps({"what": what, "kind": kind, "pid": os.getpid(), "since": since}) + "\n")
        lock.flush()
        try:
            yield
        finally:
            lock.truncate(0)  # no record outlives the hold (holder)


def holder(root: Path | None = None) -> dict[str, object] | None:
    """
    The recorded holder, if its process still lives: read without taking the lock, so a check never
    refuses a job starting at that moment. A record left by an ended job (or an old plain-text line)
    is None.
    """
    path = (runs_root() if root is None else root) / LOCK_NAME
    try:
        fields: dict[str, object] = json.loads(path.read_text())
        os.kill(int(str(fields["pid"])), 0)
    except OSError, ValueError, KeyError, TypeError:
        return None
    return fields


@contextlib.contextmanager
def study(argv: list[str] | None = None) -> Generator[None]:
    """
    A study's command line holds the machine while it trains, named by its command line (argv, or
    sys.argv's); when it's taken, the study exits with the holder's line before it trains.
    """
    words = sys.argv if argv is None else argv
    what = " ".join([Path(words[0]).name, *words[1:]]) if words else "a study"
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(hold(what, "study"))
        except MachineBusy as busy:
            sys.exit(f"not started: {busy}")
        yield
