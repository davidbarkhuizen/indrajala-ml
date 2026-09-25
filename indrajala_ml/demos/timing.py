import time
from typing import Callable, TypeVar

from indrajala_ml.train import ConvergenceSeries, train_linear_classifier_network

T = TypeVar("T")


def timed_call(fn: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
    """
    (result, elapsed_seconds) for one call, so every demo times the same span.
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def timed_train(student, train_data, learning_rate: float, epochs: int) -> tuple[ConvergenceSeries, float]:
    """
    timed_call around train_linear_classifier_network, for the demos comparing backends' training
    time.
    """
    return timed_call(train_linear_classifier_network, student, train_data, learning_rate=learning_rate, epochs=epochs)
