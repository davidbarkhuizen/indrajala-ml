"""
The hooks a network builds its layer specs and update rule from, on every implementation (the DRY
third pass workplan, D4): ArrayNetworkBase and BackpropNetworkBase list SpecHooks, and a sibling
(the ReLU, dropout, softmax and cross-entropy presets, the rule presets, the window-recurrent
network) overrides the hook it changes.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import fields
from typing import Any, ClassVar

from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay


class SpecHooks:
    """A network's default specs (sigmoid dense layers) and rule (plain SGD)."""

    def _window_state_size(self) -> int | None:
        # a window-recurrent network's state width, with which its specs are built as a window
        # (build_array_layers, build_python_layers); None for every other network
        return None

    def _hidden_spec(self, size: int) -> Dense:
        # a dense hidden layer; the ReLU and dropout siblings return theirs
        return Dense(size)

    def _output_spec(self, size: int) -> Dense:
        # the output layer; the softmax and cross-entropy siblings return theirs
        return Dense(size, output=True)

    def _dense_specs(self, layer_sizes: Sequence[int], output_size: int) -> list[LayerSpec]:
        return [*(self._hidden_spec(size) for size in layer_sizes), self._output_spec(output_size)]

    def _update_rule(self) -> UpdateRule:
        # plain SGD; the momentum, Adam and L2 siblings return their rule, from their hyperparameters
        return SGD()


class GivenRule:
    """A network built with its update rule as an argument (the Sequential and window-recurrent networks)."""

    update_rule: UpdateRule

    def _update_rule(self) -> UpdateRule:
        return self.update_rule


class RulePreset:
    """
    A preset that trains with one update rule, from hyperparameters it stores under the rule's field
    names (the DRY third pass workplan, D3): `rule` names the rule, `hyperparameters` (which the
    format-2 file and the legacy envelope record) are its dataclass fields, and _update_rule builds
    it from them. Listed first, ahead of the preset's network; the preset's own typed __init__
    stores each hyperparameter.
    """

    rule: ClassVar[type[Momentum | Adam | WeightDecay]]
    hyperparameters: ClassVar[tuple[str, ...]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # the rule's fields the preset's __init__ takes: a field added to a rule since (Adam's
        # weight_decay, the training-recipe workplan, D9) keeps its default, so the presets' files
        # and envelopes are unchanged
        taken = inspect.signature(cls.__init__).parameters
        cls.hyperparameters = tuple(f.name for f in fields(cls.rule) if f.name in taken)

    def _update_rule(self) -> UpdateRule:
        return self.rule(**{name: getattr(self, name) for name in self.hyperparameters})
