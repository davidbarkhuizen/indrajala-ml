"""
The pieces shared by the batch-size-scaling study (#365-#368): its sweep script
(scripts/batch_size_scaling_sweep.py), its timing script (scripts/batch_size_timing.py) and its
demo (demo_batch_size_scaling).

The question: does the linear learning-rate scaling rule (Goyal et al. 2017: multiply the rate
by the factor the batch grows, with warmup) hold for the dense 784 -> 30 -> 10 network on full
MNIST from batch 32 to 1024? Findings, 5 epochs, 5 seeds, Rust (the full tables are in the
study's workplan, deleted after it finished: git show 5ee017f:docs/batch-size-scaling-workplan.md):

- Best batch-32 rates: 4 at momentum 0.0 (95.44% +- 0.44%), 0.25 at momentum 0.9
  (95.13% +- 0.35%). Both sit near the stability edge (16 diverges at 0.0; 4 at 0.9). The demos'
  0.5 without momentum reaches only 93.61%.
- Without warmup the scaled rate diverges from batch 128 up, at both momenta.
- With warmup (linear_warmup, set in epochs and converted to steps) the rule holds to B = 128 at
  momentum 0.0 and to B = 512 at momentum 0.9, with a 1-epoch warmup only (95.09% +- 0.09%, 16x
  fewer steps). Past that it fails: at momentum 0.0, rates 64 and 128 stay at chance whatever the
  warmup. The likely cause, untested: a curvature ceiling on the stable rate, which a larger
  batch cannot raise.
- The unscaled control falls further behind as B grows, but where the scaled rate diverges it is
  the better choice by up to 80 points.
- Momentum 0.9 holds the rule to a 4x larger batch than 0.0, but its effective rate
  (0.25 / (1 - 0.9) = 2.5) is lower than 4, so momentum and rate are confounded.
- Warmup costs nothing at batch 32. Momentum 0.9 at the unscaled rate is unstable in early
  epochs at larger batches, which a 1-epoch warmup removes. This is unexplained.

Pending a rerun: the momentum 0.9 findings with warmup were measured with Rumelhart et al.'s
momentum (Goyal et al.'s eq. (10), without the momentum correction a changing rate needs). The
momentum layers now follow eq. (9) (README, Update rules), which differs from it only while the rate
changes. The cells without warmup change only by rounding.

The timing findings are in docs/optimizations/ (current-baseline.md and candidates.md).

The study runs its own epoch loop rather than train_backprop_network_mini_batch. The loop is the
trainer's (reshuffle every epoch, chunk into batches, one learn_batch per batch, the schedule
indexed by batch step across epochs) minus the pocket snapshot: the trainer's pass over the
training set before and after every epoch would dominate a full-MNIST epoch, and restoring the
best epoch would hide exactly the divergence the study looks for.
"""

import math
import random
import time
from collections.abc import Callable, Sequence
from typing import Any

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.lr_schedule import linear_warmup
from indrajala_ml.model.classifier_protocols import BatchTrainableClassifier, Example
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
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

SIDE = 28
DIMENSION = SIDE * SIDE
CLASS_COUNT = 10
LAYER_SIZES = [30]  # the architecture every MNIST demo uses
CONV_SPECS = [ConvSpec(3, 8)]  # the conv demo's (demo_conv_rust_vs_vectorized_digit_recognition)
CONV_DENSE_LAYER_SIZES = [32]
ARCHITECTURES = ["dense", "conv"]
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


def initial_network(backend: str, momentum: float, seed: int, architecture: str = "dense"):
    """
    A fresh network whose weights are drawn by numpy from `seed`, so every backend, rate, batch
    size and momentum sees the same starting weights for a given seed (the Rust RNG isn't
    comparable to numpy's).
    """
    if architecture == "conv":
        return _initial_conv_network(backend, momentum, seed)
    if architecture != "dense":
        raise ValueError(f"unknown architecture {architecture!r}")

    np.random.seed(seed)
    snapshot = VectorizedMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT).snapshot()

    if backend == "numpy":
        if momentum:
            network = MomentumVectorizedMultiClassBackpropClassifierNetwork(
                LAYER_SIZES, DIMENSION, CLASS_COUNT, momentum
            )
        else:
            network = VectorizedMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT)
        network.restore(snapshot)
    elif backend == "rust":
        if momentum:
            network = MomentumRustArrayMultiClassBackpropClassifierNetwork(
                LAYER_SIZES, DIMENSION, CLASS_COUNT, momentum
            )
        else:
            network = RustArrayMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT)
        network.restore([(pa.Array(W.tolist()), pa.Array(b.tolist())) for W, b in snapshot])
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return network


def _initial_conv_network(backend: str, momentum: float, seed: int):
    # no conv network has momentum (the workplan's stage 3 would add one)
    if momentum:
        raise ValueError("the conv networks have no momentum")
    np.random.seed(seed)
    snapshot = ConvVectorizedMultiClassBackpropClassifierNetwork.randomized(
        SIDE, SIDE, CONV_SPECS, CONV_DENSE_LAYER_SIZES, CLASS_COUNT
    ).snapshot()

    if backend == "numpy":
        network_cls = ConvVectorizedMultiClassBackpropClassifierNetwork
    elif backend == "rust":
        network_cls = ConvRustArrayMultiClassBackpropClassifierNetwork
    else:
        raise ValueError(f"unknown backend {backend!r}")
    network = network_cls(SIDE, SIDE, CONV_SPECS, CONV_DENSE_LAYER_SIZES, CLASS_COUNT)
    network.restore(snapshot)
    return network


def train_epoch(
    network: BatchTrainableClassifier[int],
    train_data: Sequence[Example[int]],
    batch_size: int,
    learning_rate: float | Callable[[int], float],
    first_step: int,
) -> tuple[int, float]:
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
    train_data: Sequence[Example[int]],
    test_data: Sequence[Example[int]],
    batch_size: int,
    rate: float,
    warmup_epochs: float,
    momentum: float,
    epochs: int,
    seed: int,
    architecture: str = "dense",
) -> dict[str, Any]:
    """
    One run: test accuracy after every epoch, steps per epoch and step-loop seconds per epoch.
    """
    network = initial_network(backend, momentum, seed, architecture)
    random.seed(seed)  # the shuffle order
    schedule = learning_rate_schedule(rate, warmup_steps(warmup_epochs, len(train_data), batch_size))

    test_accuracies: list[float] = []
    step_seconds: list[float] = []
    step = 0
    for _ in range(epochs):
        steps, seconds = train_epoch(network, train_data, batch_size, schedule, step)
        step += steps
        step_seconds.append(seconds)
        test_accuracies.append(accuracy(network, test_data))
    return {"test_accuracies": test_accuracies, "steps": step, "step_seconds": step_seconds}
