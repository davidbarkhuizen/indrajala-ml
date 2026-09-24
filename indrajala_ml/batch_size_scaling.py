"""
The pieces shared by the batch-size-scaling study (docs/batch-size-scaling-workplan.md): its
sweep script, its timing script and its demo.

The study runs its own epoch loop rather than train_backprop_network_mini_batch. The loop is the
trainer's (reshuffle every epoch, chunk into batches, one learn_batch per batch, the schedule
indexed by batch step across epochs) minus the pocket snapshot: the trainer's pass over the
training set before and after every epoch would dominate a full-MNIST epoch, and restoring the
best epoch would hide exactly the divergence the study looks for.
"""

import math
import random
import time
from typing import Callable

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.lr_schedule import linear_warmup
from indrajala_ml.model.momentum_rust_array_multiclass_backprop_classifier_network import (
    MomentumRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_vectorized_multiclass_backprop_classifier_network import (
    MomentumVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.multiclass_evaluate import accuracy

DIMENSION = 28 * 28
CLASS_COUNT = 10
LAYER_SIZES = [30]  # the architecture every MNIST demo uses
BASE_BATCH_SIZE = 32
TRAIN_PATH = "data/mnist/mnist-train.bin"
TEST_PATH = "data/mnist/mnist-test.bin"


def scaled_learning_rate(base_rate: float, batch_size: int) -> float:
    # the linear scaling rule: learn_batch averages the gradient over the batch, so growing the
    # batch by a factor k takes k times fewer steps of the same expected size unless the rate
    # grows by k too
    return base_rate * batch_size / BASE_BATCH_SIZE


def warmup_steps(warmup_epochs: float, train_size: int, batch_size: int) -> int:
    # warmup is set in epochs, so it covers the same share of training at every batch size
    return math.ceil(warmup_epochs * train_size / batch_size)


def learning_rate_schedule(rate: float, warmup_step_count: int) -> float | Callable[[int], float]:
    return linear_warmup(rate, warmup_step_count) if warmup_step_count > 0 else rate


def initial_network(backend: str, momentum: float, seed: int):
    """
    A fresh network whose weights are drawn by numpy from `seed`, so every backend, rate, batch
    size and momentum sees the same starting weights for a given seed (the Rust RNG isn't
    comparable to numpy's).
    """
    np.random.seed(seed)
    snapshot = VectorizedMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT).snapshot()

    if backend == "numpy":
        if momentum:
            network = MomentumVectorizedMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT, momentum)
        else:
            network = VectorizedMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT)
        network.restore(snapshot)
    elif backend == "rust":
        if momentum:
            network = MomentumRustArrayMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT, momentum)
        else:
            network = RustArrayMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT)
        network.restore([(pa.Array(W.tolist()), pa.Array(b.tolist())) for W, b in snapshot])
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return network


def train_epoch(network, train_data: list, batch_size: int, learning_rate, first_step: int) -> tuple[int, float]:
    """
    One epoch of the trainer's loop: reshuffle (from the caller's random state), then one
    learn_batch per batch. Returns (steps taken, seconds spent in learn_batch calls); the
    shuffle and batch slicing are outside the timed span.
    """
    epoch_data = list(train_data)
    random.shuffle(epoch_data)
    batches = [epoch_data[i : i + batch_size] for i in range(0, len(epoch_data), batch_size)]

    step_seconds = 0.0
    for step, batch in enumerate(batches, start=first_step):
        rate = learning_rate(step) if callable(learning_rate) else learning_rate
        start = time.perf_counter()
        network.learn_batch(rate, batch)
        step_seconds += time.perf_counter() - start
    return len(batches), step_seconds


def train_and_evaluate(
    backend: str,
    train_data: list,
    test_data: list,
    batch_size: int,
    rate: float,
    warmup_epochs: float,
    momentum: float,
    epochs: int,
    seed: int,
) -> dict:
    """
    One run: test accuracy after every epoch, steps per epoch and step-loop seconds per epoch.
    """
    network = initial_network(backend, momentum, seed)
    random.seed(seed)  # the shuffle order
    schedule = learning_rate_schedule(rate, warmup_steps(warmup_epochs, len(train_data), batch_size))

    test_accuracies = []
    step_seconds = []
    step = 0
    for _ in range(epochs):
        steps, seconds = train_epoch(network, train_data, batch_size, schedule, step)
        step += steps
        step_seconds.append(seconds)
        test_accuracies.append(accuracy(network, test_data))
    return {"test_accuracies": test_accuracies, "steps": step, "step_seconds": step_seconds}
