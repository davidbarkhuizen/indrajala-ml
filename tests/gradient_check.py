"""
A finite-difference gradient check for any Sequential network in any implementation (the
batch-norm workplan, stage 0): the gradient a network accumulates over one batch, through its own
learn_batch, against central differences of the whole batch's loss.

The loss is a function of the whole batch, not a sum of per-example checks, so a layer whose
forward pass depends on the rest of the batch (batch norm) is checked as it trains. Every loss
evaluation first restores the network's snapshot, so a layer's state that the training forward
pass moves (batch norm's running averages, which snapshot() carries) is the same for each
perturbation, not a moving target.

The analytical gradient is read from the network's real training step: the network's optimizer is
swapped for one that records each layer's accumulated gradients and resets them, instead of
stepping. The update rule doesn't matter, since the rule only consumes the gradient.

    check_gradients(network, batch)  # raises GradientMismatch naming the worst weight

A network that drops out draws new masks each pass, so its check takes rng, a factory of fresh
generators: every pass, the analytic one and each loss evaluation, starts from rng(), so every pass
drops the same units and the loss is a function of the weights alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

from indrajala_ml.model.networks.array_network_base import ArrayNetworkBase
from indrajala_ml.model.networks.python.backprop_network_base import BackpropNetworkBase
from indrajala_ml.model.specs.layer_specs import Dense, token_wise_output

# a batch's loss from its output rows and target rows
Loss = Callable[[list[list[float]], list[list[float]]], float]

# central differences' step: about the cube root of float64's epsilon, where truncation and
# rounding error balance
STEP = 1e-6
# |numeric - analytic| <= ABSOLUTE + RELATIVE * |analytic|: on today's layers the finite
# difference is within 1e-9 of the analytic gradient (tests/test_gradient_check.py), ten times
# under ABSOLUTE, while a wrong term in a backward pass is off by about the gradient itself
ABSOLUTE = 1e-8
RELATIVE = 1e-6


def squared_loss(outputs: list[list[float]], targets: list[list[float]]) -> float:
    # the sigmoid output layer's delta, (a - t) * a * (1 - a), is this loss's gradient
    return sum(0.5 * (a - t) * (a - t) for row, target in zip(outputs, targets) for a, t in zip(row, target))


def binary_cross_entropy_loss(outputs: list[list[float]], targets: list[list[float]]) -> float:
    # per node, the cross-entropy sigmoid output layer's delta, a - t, is this loss's gradient
    return -sum(
        t * math.log(a) + (1.0 - t) * math.log(1.0 - a)
        for row, target in zip(outputs, targets)
        for a, t in zip(row, target)
    )


def softmax_cross_entropy_loss(outputs: list[list[float]], targets: list[list[float]]) -> float:
    # the softmax output layer's delta, a - t, is this loss's gradient
    return -sum(t * math.log(a) for row, target in zip(outputs, targets) for a, t in zip(row, target))


def token_cross_entropy_loss(tokens: int) -> Loss:
    # a token-wise softmax output layer's delta, (a - t) / T, is the gradient of the mean of its
    # tokens' cross-entropies (the sequence task workplan, D6)
    def loss(outputs: list[list[float]], targets: list[list[float]]) -> float:
        return softmax_cross_entropy_loss(outputs, targets) / tokens

    return loss


def loss_of(output: Dense, tokens: int | None = None) -> Loss:
    """The loss whose gradient the output layer's delta is; tokens for a token-wise one."""
    if tokens is not None:
        return token_cross_entropy_loss(tokens)
    if output.activation == "softmax":
        return softmax_cross_entropy_loss
    return binary_cross_entropy_loss if output.loss == "cross_entropy" else squared_loss


def targets(network: Any, labels: Sequence[Any]) -> list[list[float]]:
    # one-hot rows for a multiclass network, the label itself for a one-output network, and one-hot
    # per token, side by side, for a sequence network's labels, one class per token
    size = network.layer_specs[-1].size
    if labels and isinstance(labels[0], tuple):
        return [[1.0 if i == token else 0.0 for token in label for i in range(size)] for label in labels]
    if size == 1:
        return [[float(label)] for label in labels]
    return [[1.0 if i == label else 0.0 for i in range(size)] for label in labels]


class _RecordingOptimizer:
    """
    Stands in for a network's optimizer during one learn_batch: it records each layer's accumulated
    gradients, as nested lists shaped as the layer's snapshot entry, and resets them.
    """

    def __init__(self, python: bool) -> None:
        # a pure-Python network's layers all have weight_sets(); an array layer has gradients(),
        # or nothing to train (a pool layer)
        self.python = python
        self.gradients: dict[int, Any] = {}

    def begin_step(self) -> None:
        pass

    def apply(self, index: int, layer: Any, _learning_rate: float, _batch_size: int) -> None:
        if self.python:
            # a dense layer's nodes, a conv layer's kernels or a batch-norm layer's features ([gamma]
            # and beta), as snapshot_state, a linear layer's nodes without a bias; none for pool
            self.gradients[index] = [
                [list(weight_set.weight_gradient_accum), weight_set.bias_gradient_accum]
                if weight_set.has_bias
                else [list(weight_set.weight_gradient_accum)]
                for weight_set in layer.weight_sets()
            ]
            for weight_set in layer.weight_sets():
                weight_set.reset_gradient_accum()
        elif hasattr(layer, "gradients"):
            # TrainedArrayLayer: (grad_W, grad_b), a linear layer's (grad_W,), batch norm's
            # (grad_gamma, grad_beta)
            self.gradients[index] = [gradient.tolist() for gradient in layer.gradients()]
            layer.reset_gradient_accum()
        else:
            self.gradients[index] = []  # a pool layer

    def step_single(self, *_args: Any) -> None:
        raise AssertionError("the gradient check trains through learn_batch only")


def analytic_gradients(network: Any, states: Sequence[tuple[float, ...]], labels: Sequence[Any]) -> list[Any]:
    """
    The gradients the network accumulates over the batch in one learn_batch, summed over the
    batch (not divided by its size), per layer in snapshot order. The weights don't move.
    """
    recorder = _RecordingOptimizer(isinstance(network, BackpropNetworkBase))
    optimizer = network.optimizer
    network.optimizer = recorder
    try:
        network.learn_batch(1.0, list(zip(states, labels)))
    finally:
        network.optimizer = optimizer
    return [recorder.gradients[index] for index in range(len(recorder.gradients))]


def training_outputs(network: Any, states: Sequence[tuple[float, ...]]) -> list[list[float]]:
    """The output rows of a training-mode forward pass over the batch, as learn_batch runs it."""
    network._set_training_mode(True)
    try:
        if isinstance(network, ArrayNetworkBase):
            array_network = cast("ArrayNetworkBase[Any]", network)
            batch = array_network.backend.matrix(states)
            for layer in array_network.layers:
                batch = layer.forward_batch(batch)
            return batch.tolist()
        # one example at a time, or layer-major for a network with batch norm or dropout, as _learn_batch runs it
        return network._forward_batch_outputs(states)
    finally:
        network._set_training_mode(False)


def _as_lists(value: Any) -> Any:
    # a snapshot as nested lists (arrays through tolist, tuples as lists), so one weight can be set
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list | tuple):
        return [_as_lists(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def _leaves(tree: Any, path: tuple[int, ...] = ()) -> Iterator[tuple[tuple[int, ...], float]]:
    if isinstance(tree, list):
        for i, item in enumerate(tree):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            yield from _leaves(item, (*path, i))
    else:
        yield path, float(tree)


def _with_value(tree: Any, path: tuple[int, ...], value: float) -> Any:
    # a copy of tree with the leaf at path set to value
    if not path:
        return value
    head, *rest = path
    copied = list(tree)
    copied[head] = _with_value(copied[head], tuple(rest), value)
    return copied


@dataclass(frozen=True)
class Comparison:
    # a weight's path in the snapshot (layer, then the entry's indices), its two gradients, and
    # how far apart they are in units of the allowed error
    path: tuple[int, ...]
    analytic: float
    numeric: float

    @property
    def excess(self) -> float:
        return abs(self.numeric - self.analytic) / (ABSOLUTE + RELATIVE * abs(self.analytic))


class GradientMismatch(AssertionError):
    pass


def compare_gradients(
    network: Any,
    states: Sequence[tuple[float, ...]],
    labels: Sequence[Any],
    loss: Loss | None = None,
    rng: Callable[[], Any] | None = None,
) -> list[Comparison]:
    """
    Every trained weight's analytic gradient and its central difference, for the whole batch's
    loss (loss_of the output spec, unless given), each pass from rng() when given. Leaves the
    network's weights as they were.
    """
    specs = network.layer_specs
    tokens = getattr(network, "tokens", None) if token_wise_output(specs) else None
    loss = loss_of(specs[-1], tokens) if loss is None else loss
    target_rows = targets(network, labels)
    base = _as_lists(network.snapshot())

    def reseed() -> None:
        if rng is not None:
            network.rng = rng()

    def batch_loss(snapshot: Any) -> float:
        network.restore(snapshot)
        reseed()
        return loss(training_outputs(network, states), target_rows)

    try:
        network.restore(base)
        reseed()
        gradients = analytic_gradients(network, states, labels)
        comparisons: list[Comparison] = []
        # a layer's snapshot entry may hold state after its trained weights (batch norm's running
        # averages), which has no gradient: the gradient tree's leaves are the weights
        for path, analytic in _leaves(gradients):
            weight = _leaves_at(base, path)
            plus = batch_loss(_with_value(base, path, weight + STEP))
            minus = batch_loss(_with_value(base, path, weight - STEP))
            comparisons.append(Comparison(path, analytic, (plus - minus) / (2 * STEP)))
    finally:
        network.restore(base)
    return comparisons


def _leaves_at(tree: Any, path: tuple[int, ...]) -> float:
    for i in path:
        tree = tree[i]
    return float(tree)


def check_gradients(
    network: Any,
    states: Sequence[tuple[float, ...]],
    labels: Sequence[Any],
    loss: Loss | None = None,
    rng: Callable[[], Any] | None = None,
) -> list[Comparison]:
    """
    compare_gradients, raising GradientMismatch at the worst weight if any is outside the allowed
    error, or if every gradient is zero (a check that can't fail). Returns the comparisons.
    """
    comparisons = compare_gradients(network, states, labels, loss, rng)
    assert comparisons, "the network has no trained weights to check"
    assert any(c.analytic != 0.0 for c in comparisons), "every analytic gradient is zero: the check would be vacuous"
    worst = max(comparisons, key=lambda c: c.excess)
    if worst.excess > 1.0:
        raise GradientMismatch(
            f"{sum(c.excess > 1.0 for c in comparisons)} of {len(comparisons)} weights mismatch; worst at "
            f"{worst.path}: analytic {worst.analytic!r}, numeric {worst.numeric!r}"
        )
    return comparisons
