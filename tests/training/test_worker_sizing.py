import pickle

import pytest

from indrajala_ml.training.worker_sizing import available_memory_bytes, estimate_bytes_per_example, select_worker_count
from tests.helpers import approx


def test_available_memory_bytes_returns_a_real_positive_value_on_linux():

    # this dev environment is Linux (see cli's install_os_packages), so /proc/meminfo should
    # genuinely exist and parse to a plausible value, not just "doesn't crash"
    available = available_memory_bytes()

    assert available is not None
    assert available > 0


def test_estimate_bytes_per_example_matches_a_direct_pickle_measurement():

    dataset = [((float(i), float(i) * 2), i % 3) for i in range(200)]

    estimated = estimate_bytes_per_example(dataset, sample_size=50)

    direct = len(pickle.dumps(dataset[:50])) / 50
    assert estimated == approx(direct)


def test_estimate_bytes_per_example_handles_a_dataset_smaller_than_the_sample_size():

    dataset = [((1.0, 2.0), 0), ((3.0, 4.0), 1)]

    estimated = estimate_bytes_per_example(dataset, sample_size=50)

    assert estimated > 0


def test_estimate_bytes_per_example_rejects_an_empty_dataset():

    with pytest.raises(AssertionError):
        estimate_bytes_per_example([], sample_size=50)


def test_select_worker_count_is_limited_by_available_memory(monkeypatch: pytest.MonkeyPatch):

    # half of 100MB for workers at 1000 examples * 1000 bytes * 2.0 safety = 2MB each: 25
    # workers. cpu_count and class_count are set high enough not to bind
    monkeypatch.setattr("indrajala_ml.training.worker_sizing.available_memory_bytes", lambda: 100_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 25


def test_select_worker_count_is_limited_by_cpu_count_when_memory_is_abundant(monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setattr("indrajala_ml.training.worker_sizing.available_memory_bytes", lambda: 10_000_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 4)

    worker_count = select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 4


def test_select_worker_count_is_limited_by_class_count(monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setattr("indrajala_ml.training.worker_sizing.available_memory_bytes", lambda: 10_000_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = select_worker_count(
        class_count=3, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 3


def test_select_worker_count_respects_an_explicit_lower_request(monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setattr("indrajala_ml.training.worker_sizing.available_memory_bytes", lambda: 10_000_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=2
    )

    assert worker_count == 2


def test_select_worker_count_never_goes_below_one_even_under_severe_memory_pressure(monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setattr("indrajala_ml.training.worker_sizing.available_memory_bytes", lambda: 1)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = select_worker_count(
        class_count=64,
        estimated_examples_per_classifier=1_000_000,
        bytes_per_example=1000.0,
        requested_worker_count=None,
    )

    assert worker_count == 1


def test_select_worker_count_falls_back_to_cpu_and_class_count_when_memory_is_undetectable(
    monkeypatch: pytest.MonkeyPatch,
):

    monkeypatch.setattr("indrajala_ml.training.worker_sizing.available_memory_bytes", lambda: None)
    monkeypatch.setattr("os.cpu_count", lambda: 4)

    worker_count = select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 4
