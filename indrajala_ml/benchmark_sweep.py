import multiprocessing
import os
import statistics
import time
from typing import Any, Callable, Hashable

_worker_fn: Callable[[Any, Hashable, int], Any] | None = None
_shared_context: Any = None


def _init_worker(worker_fn: Callable[[Any, Hashable, int], Any], shared_context: Any) -> None:
    # runs once per worker process, not once per job - worker_fn and shared_context are both
    # pickled through Pool's own initargs a single time per worker, not re-pickled on every
    # (config, seed) task the way passing them through imap's own per-job arguments would.
    global _worker_fn, _shared_context
    _worker_fn = worker_fn
    _shared_context = shared_context


def _run_one_job(job: tuple[Hashable, int]) -> tuple[Hashable, int, Any]:
    config, seed = job
    assert _worker_fn is not None, "_run_one_job called before a Pool initializer set _worker_fn"
    return config, seed, _worker_fn(_shared_context, config, seed)


def run_parameter_sweep(
    configs: list[Hashable],
    seeds: list[int],
    worker_fn: Callable[[Any, Hashable, int], Any],
    shared_context: Any = None,
    worker_count: int | None = None,
    report_progress: bool = True,
) -> dict[Hashable, list[Any]]:
    """
    A multiprocessing.Pool sweep runner. Dispatches one job per (config, seed) pair;
    worker_fn(shared_context, config, seed) must be a module-level function
    (not a local closure or lambda), the same picklability constraint ensemble_train.py's own
    worker functions already have, for the same reason - Pool workers receive tasks through a
    queue, which pickles whatever callable and arguments each task carries.

    shared_context is whatever read-only context every job needs in common (e.g. a
    benchmark_data.BenchmarkProxy built once by the caller before this call, or None if a worker
    needs no shared context at all) - passed explicitly through Pool's own initializer/initargs
    alongside worker_fn itself, pickled once per worker process, not per job. Deliberately
    explicit rather than the alternative of the caller stashing shared_context in a module-level
    global for worker_fn to read: that would only actually work under multiprocessing's fork
    start method (Linux's default, where a forked child inherits a copy of the parent's
    already-populated globals), silently breaking under spawn (macOS/Windows' default, where a
    spawned child re-imports the module fresh with none of the parent's mutations) - the same
    "pass every worker's context through explicit task/init arguments, never a fork-inherited
    global" idiom ensemble_train.py's own Pool workers already use, for exactly this portability
    reason.

    Each job receives a seed derived directly from `seeds` (not further reseeded from a parent
    random.Random the way ensemble_train.py's per-class jobs are, since here the caller already
    controls the exact seed list a sweep runs) - `worker_fn` itself is responsible for calling
    `random.seed(seed)`/`np.random.seed(seed)` as its own training path requires, exactly as
    every existing sweep script already does by hand.

    Progress is printed with flush=True as each job completes (when report_progress is True) -
    the concrete fix for a real gap found while running dropout's own stage-3 sweep: Python fully
    buffers stdout when it isn't a tty, so a per-job print() during a long run launched via
    run_in_background and redirected to a file would otherwise queue silently instead of showing
    interim progress.
    """

    assert len(configs) > 0, "configs must be non-empty"
    assert len(seeds) > 0, "seeds must be non-empty"

    resolved_worker_count = worker_count or os.cpu_count() or 1
    jobs = [(config, seed) for config in configs for seed in seeds]
    results: dict[Hashable, list[Any]] = {config: [] for config in configs}

    with multiprocessing.Pool(
        resolved_worker_count, initializer=_init_worker, initargs=(worker_fn, shared_context)
    ) as pool:
        for completed, (config, _seed, result) in enumerate(pool.imap(_run_one_job, jobs), start=1):
            results[config].append(result)
            if report_progress:
                print(f"{completed}/{len(jobs)} jobs complete", flush=True)

    return results


def estimate_sweep_wallclock(
    worker_fn: Callable[[Any, Hashable, int], Any],
    sample_config: Hashable,
    sample_seed: int,
    planned_run_count: int,
    shared_context: Any = None,
    worker_count: int | None = None,
) -> float:
    """
    Runs one real job serially and projects the full sweep's total wall-clock, replacing the
    informal calibration step every existing sweep in this codebase's docs has done by hand
    before committing to a full run (e.g. dropout's own stage-3 calibration: "a real, timed
    calibration run... took 0.85s, projecting..."). Not a substitute for actually running the
    sweep and measuring it directly - a projection, exactly as informal calibration always was,
    just no longer requiring a human to do the arithmetic by hand each time.

    worker_fn/shared_context have the same shape run_parameter_sweep's own do, so a caller can
    pass the identical worker_fn and shared_context to both calls.
    """

    resolved_worker_count = worker_count or os.cpu_count() or 1

    start = time.perf_counter()
    worker_fn(shared_context, sample_config, sample_seed)
    single_job_seconds = time.perf_counter() - start

    return single_job_seconds * planned_run_count / resolved_worker_count


def summarize_sweep_results(results: dict[Hashable, list[float]], value_format: str = ".2%") -> str:
    """
    Renders mean/stdev per config as a markdown table, in the same shape every research doc's
    own results table already uses. Multi-dimensional tables (e.g. the batch-size sweeps, one
    row per config and one column per batch size) remain the caller's own assembly - this covers
    the common one-result-per-(config, seed) shape, not every table layout this codebase's docs
    have used.
    """

    lines = ["| config | mean | stdev |", "|---|---|---|"]
    for config, values in results.items():
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(f"| {config} | {format(mean, value_format)} | {format(stdev, value_format)} |")

    return "\n".join(lines)
