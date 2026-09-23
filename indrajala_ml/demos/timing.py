import time
from typing import Callable, TypeVar

from indrajala_ml.train import ConvergenceSeries, train_linear_classifier_network

T = TypeVar("T")


def timed_call(fn: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
    """
    Shared by every demo that reports a backend's wall-clock cost alongside its result
    (demo_vectorized_mnist_recognition.py's own MNIST-decode comparison, on top of the
    training-cost use timed_train below wraps) - times a single call, returns
    (result, elapsed_seconds), so the timing methodology (what's inside vs. outside the timed
    span) can't silently drift between call sites the way it can when each one hand-writes its
    own perf_counter()-before/after pair.
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def timed_train(student, train_data, learning_rate: float, epochs: int) -> tuple[ConvergenceSeries, float]:
    """
    The training-specific case timed_call above generalizes - shared by every demo comparing
    multiple network backends' wall-clock training cost
    (demo_vectorized_uci_digit_recognition.py, demo_vectorized_mnist_recognition.py,
    demo_rust_vs_vectorized_uci_digit_recognition.py,
    demo_rust_vs_vectorized_mnist_recognition.py), each of which independently wrapped
    train_linear_classifier_network in an identical timing pair before this existed.
    """
    return timed_call(train_linear_classifier_network, student, train_data, learning_rate=learning_rate, epochs=epochs)
