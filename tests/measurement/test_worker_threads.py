"""indrajala_ml/measurement/worker_threads.py: a pool's workers get their budget, in every start method, and a pool past the host's CPUs is refused."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import indrajala_math_rust
import pytest

from indrajala_ml.measurement.worker_threads import THREADS_VARIABLE, declared_threads, worker_threads


def budget(_: int = 0) -> int:
    return indrajala_math_rust.threads()


@pytest.mark.parametrize(("value", "expected"), [(None, 1), ("3", 3), (" 2 ", 2), ("0", 1), ("two", 1), ("-1", 1)])
def test_a_worker_gets_one_thread_unless_the_environment_declares_more(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: int
) -> None:
    monkeypatch.delenv(THREADS_VARIABLE, raising=False)
    if value is not None:
        monkeypatch.setenv(THREADS_VARIABLE, value)
    assert declared_threads() == expected


def test_a_declared_count_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(THREADS_VARIABLE, "3")
    assert declared_threads(2) == 2


@pytest.mark.parametrize(("workers", "threads", "fits"), [(8, 1, True), (4, 2, True), (9, 1, False), (3, 3, False)])
def test_a_pool_past_the_host_s_cpus_is_refused(workers: int, threads: int, fits: bool) -> None:
    if fits:
        assert worker_threads(workers, threads, cpus=8)["initargs"] == (threads,)
    else:
        with pytest.raises(ValueError, match="more than this host's 8 CPUs"):
            worker_threads(workers, threads, cpus=8)


@pytest.mark.parametrize("method", ["forkserver", "spawn", "fork"])
def test_every_start_method_s_workers_get_the_budget(method: str) -> None:
    context = multiprocessing.get_context(method)
    with ProcessPoolExecutor(1, mp_context=context, **worker_threads(1, 3, cpus=3)) as pool:
        assert pool.submit(budget).result() == 3
