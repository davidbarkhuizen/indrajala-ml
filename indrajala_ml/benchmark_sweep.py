import multiprocessing
import os
import statistics
import time
from typing import Any, Callable, Hashable

_worker_fn: Callable[[Any, Hashable, int], Any] | None = None
_shared_context: Any = None


def _init_worker(worker_fn: Callable[[Any, Hashable, int], Any], shared_context: Any) -> None:
    # once per worker process: worker_fn and shared_context are pickled once per worker through
    # initargs, not once per job
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
    Runs worker_fn(shared_context, config, seed) for every (config, seed) pair on a
    multiprocessing.Pool, returning each config's results in seed order. worker_fn must be a
    module-level function: Pool pickles it.

    shared_context is read-only data every job needs (e.g. a BenchmarkProxy), or None. It is passed
    through the Pool initializer, pickled once per worker, rather than left in a module global,
    which only fork (Linux) would carry into workers and spawn (macOS, Windows) would not.

    Each job gets its seed from `seeds` as is; worker_fn seeds random/np.random itself.

    With report_progress, a "completed/total" line is printed and flushed per finished job: stdout
    redirected to a file is fully buffered, which would hide progress on a long background run.
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
    Times one job serially and projects the whole sweep's wall-clock from it: a projection, not a
    measurement. worker_fn/shared_context are run_parameter_sweep's, so a caller can pass the same
    ones to both.
    """

    resolved_worker_count = worker_count or os.cpu_count() or 1

    start = time.perf_counter()
    worker_fn(shared_context, sample_config, sample_seed)
    single_job_seconds = time.perf_counter() - start

    return single_job_seconds * planned_run_count / resolved_worker_count


def summarize_sweep_results(results: dict[Hashable, list[float]], value_format: str = ".2%") -> str:
    """
    Mean and stdev per config as a markdown table, for one result per (config, seed). Other layouts
    (e.g. one column per batch size) are the caller's.
    """

    lines = ["| config | mean | stdev |", "|---|---|---|"]
    for config, values in results.items():
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(f"| {config} | {format(mean, value_format)} | {format(stdev, value_format)} |")

    return "\n".join(lines)
