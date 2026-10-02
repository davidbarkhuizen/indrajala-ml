"""
What the three optimizers share (python_optimizer.py, numpy_optimizer.py and rust_optimizer.py): the
update rule, the step count t that begin_step() advances, the rule's state per layer, and the
choice, made once at construction, of the subclass's _apply_* method that steps a layer under the
rule. The formulas are each subclass's own.

Backend-free, so the pure-Python optimizer imports it too.
"""

from __future__ import annotations

from collections.abc import Callable

from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay


class OptimizerBase[S, T, R]:
    """
    S is the rule's state for one layer; T what an _apply_* method steps (a layer, or a pure-Python
    layer's weight sets); R what it returns (the stepped parameters on Rust, which rebinds them, and
    None elsewhere).
    """

    def __init__(self, rule: UpdateRule) -> None:
        self.rule = rule
        self.t = 0
        # per layer index: the rule's state, made on the layer's first step
        self._state: dict[int, S] = {}
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

    # per subclass: each rule's step of one layer, whose index keys its state
    def _apply_sgd(self, index: int, layer: T, learning_rate: float, batch_size: int, /) -> R:
        raise NotImplementedError

    def _apply_momentum(self, index: int, layer: T, learning_rate: float, batch_size: int, /) -> R:
        raise NotImplementedError

    def _apply_adam(self, index: int, layer: T, learning_rate: float, batch_size: int, /) -> R:
        raise NotImplementedError

    def _apply_weight_decay(self, index: int, layer: T, learning_rate: float, batch_size: int, /) -> R:
        raise NotImplementedError
