"""
What the three optimizers share (python_optimizer.py, numpy_optimizer.py and rust_optimizer.py): the
update rule, the step count t that begin_step() advances, the rule's state per layer, the choice,
made once at construction, of the subclass's _apply_* method that steps a layer under the rule, and
step(), a batch's update of every layer once all their gradients are accumulated, with clipping by
the global norm (the training-recipe workplan, D10). The formulas are each subclass's own.

Backend-free, so the pure-Python optimizer imports it too.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay


class OptimizerBase[S, T, R, L]:
    """
    S is the rule's state for one layer; T what an _apply_* method steps (a layer, or a pure-Python
    layer's weight sets); R what it returns (the stepped parameters on Rust, which rebinds them, and
    None elsewhere); L a network's layer, as step() and apply() take it.
    """

    def __init__(self, rule: UpdateRule) -> None:
        self.rule = rule
        self.t = 0
        # per layer index: the rule's state, made on the layer's first step
        self._state: dict[int, S] = {}
        # the factor every gradient is read times during a step that clips (step()), None otherwise:
        # each subclass's _apply_* methods read their gradients through it
        self._gradient_scale: float | None = None
        self._apply_rule: Callable[[int, T, float, int], R]
        match rule:
            case SGD():
                self._apply_rule = self._apply_sgd
            case Momentum():
                self._apply_rule = self._apply_momentum
            case Adam():
                self._apply_rule = self._apply_adam
            case WeightDecay():
                self._apply_rule = self._apply_weight_decay

    def begin_step(self) -> None:
        self.t += 1

    def step(self, layers: Sequence[L], learning_rate: float, batch_size: int, clip_norm: float | None = None) -> None:
        """
        One batch's update of every layer, in order, its gradients accumulated over batch_size
        examples: the step count advances, then each layer is applied under the rule (apply(), its
        index in layers keying its state). With clip_norm, when the batch gradient's global L2 norm
        ||g|| (every layer's accumulated gradients over batch_size, a left fold over the layers in
        order, their gradients in order; Pascanu et al. 2013) is above clip_norm, every gradient is
        read times clip_norm / ||g|| (D10), so the step's direction is kept and its gradient's norm
        is clip_norm.
        """
        self.begin_step()
        self._gradient_scale = None if clip_norm is None else self._clip_scale(layers, batch_size, clip_norm)
        try:
            for index, layer in enumerate(layers):
                self.apply(index, layer, learning_rate, batch_size)
        finally:
            self._gradient_scale = None

    def _clip_scale(self, layers: Sequence[L], batch_size: int, clip_norm: float) -> float | None:
        squares = 0.0
        for layer in layers:
            squares += self._squared_gradient(layer)
        norm = math.sqrt(squares) / batch_size
        return clip_norm / norm if norm > clip_norm else None

    def apply(self, index: int, layer: L, learning_rate: float, batch_size: int) -> None:
        raise NotImplementedError

    def _squared_gradient(self, layer: L) -> float:
        # the sum of squares of the layer's accumulated gradients, its gradients in order; 0 for a
        # layer with nothing trained
        raise NotImplementedError

    # per subclass: each rule's step of one layer, whose index keys its state
    def _apply_sgd(self, index: int, layer: T, learning_rate: float, batch_size: int, /) -> R:
        raise NotImplementedError

    def _apply_momentum(self, index: int, layer: T, learning_rate: float, batch_size: int, /) -> R:
        raise NotImplementedError

    def _apply_adam(self, index: int, layer: T, learning_rate: float, batch_size: int, /) -> R:
        raise NotImplementedError

    def _apply_weight_decay(self, index: int, layer: T, learning_rate: float, batch_size: int, /) -> R:
        raise NotImplementedError
