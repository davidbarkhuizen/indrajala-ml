"""
One thread budget for a pool of worker processes (the crate performance workplan's D4). Without this
every worker takes the crate's default, a thread per physical core, so a pool of them oversubscribes the
host. A pool's workers get 1 thread each, or what `threads` or `INDRAJALA_THREADS` declares, and a pool
that would ask for more than the host's CPUs is refused.

    pool = ProcessPoolExecutor(workers, mp_context=context, **worker_threads(workers))

The worker's initializer sets the budget, rather than the environment doing it: a forkserver's workers
inherit the environment of the moment the server started, not the moment the pool did.
"""

import os
from typing import Any

THREADS_VARIABLE = "INDRAJALA_THREADS"


def available_cpus() -> int:
    """The CPUs this process may run on (logical: a study's slots may count SMT's second threads)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def declared_threads(threads: int | None = None) -> int:
    """`threads`, else a positive `INDRAJALA_THREADS`, else 1."""
    if threads is not None:
        return threads
    value = os.environ.get(THREADS_VARIABLE, "").strip()
    return int(value) if value.isdecimal() and int(value) > 0 else 1


def set_worker_threads(threads: int) -> None:
    """The initializer: this process's budget, in a worker. Module-level, so a spawned worker can find it."""
    import indrajala_math_rust

    indrajala_math_rust.set_threads(threads)


def checked_threads(workers: int, threads: int | None = None, cpus: int | None = None) -> int:
    """Each worker's budget, once `workers` x it is known to fit the host."""
    declared = declared_threads(threads)
    available = available_cpus() if cpus is None else cpus
    if workers * declared > available:
        raise ValueError(
            f"{workers} workers x {declared} threads asks for more than this host's {available} CPUs "
            "(the crate performance workplan's D4): lower one"
        )
    return declared


def worker_threads(workers: int, threads: int | None = None, cpus: int | None = None) -> dict[str, Any]:
    """A pool's `initializer` and `initargs`; a pool with its own initializer calls `checked_threads` and
    `set_worker_threads` itself."""
    return {"initializer": set_worker_threads, "initargs": (checked_threads(workers, threads, cpus),)}
