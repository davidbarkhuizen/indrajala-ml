"""
Pure-Python networks of any accepted layer specs and update rule: the counterparts of
sequential_array_network.py's, from the same specs (python_layer_builder.py). A spec list and rule
build the same network in all three implementations, so the pure-Python one is the parity
reference for any array network, preset or sequential.

    network = SequentialMultiClassBackpropClassifierNetwork(
        (8, 8, 1), [Conv(3, 4), Pool(2), Dense(16), Dense(10, output=True)], Momentum(0.9)
    )
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, Self

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.backprop_network_base import randomize_fan_in_aware
from indrajala_ml.model.bounds import validate_class_count
from indrajala_ml.model.format2 import NetworkFile
from indrajala_ml.model.layer_protocols import TrainableLayer
from indrajala_ml.model.layer_specs import Dense, InputShape, LayerSpec
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.update_rules import UpdateRule


def _bounds(input_shape: InputShape, input_bounds: list[tuple[float, float]] | None) -> list[tuple[float, float]]:
    # the input layer's bounds, which only BackpropClassifierNetwork's bounds-width randomize and
    # the save files read, and a sequential network's randomize doesn't: (0, 1) per input unless
    # given, as load_network gives a preset's
    return [(0.0, 1.0)] * math.prod(input_shape) if input_bounds is None else input_bounds


class SequentialMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork[TrainableLayer]):
    """
    A multiclass network of any accepted layer specs (layer_specs.py) and update rule, with the
    multiclass network's argmax classification, one-hot targets and fan-in-aware randomize.
    class_count is the output layer's size. It saves in format 2 (format2.py), without a preset,
    and loads any pure-Python format-2 file of its shape, a preset's included.
    """

    preset_arguments: ClassVar[tuple[str, ...] | None] = None

    def __init__(
        self,
        input_shape: InputShape,
        layers: Sequence[LayerSpec],
        update_rule: UpdateRule,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        output = layers[-1] if layers else None
        assert isinstance(output, Dense), f"the last layer must be the output layer, a Dense; got {output!r}"
        validate_class_count(output.size)

        self.class_count = output.size
        # read by _update_rule, while the base builds the optimizer
        self.update_rule = update_rule

        # past the multiclass network's __init__, whose flat layer_sizes can't describe these layers,
        # to BackpropNetworkBase's
        super(MultiClassBackpropClassifierNetwork, self).__init__(
            layers, input_shape, _bounds(input_shape, input_bounds)
        )

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    @classmethod
    def _from_file(cls, file: NetworkFile) -> Self:
        return cls(file.input_shape, file.layers, file.update_rule, file.input_bounds)

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        # not the preset parent's envelope, which no Sequential network ever wrote
        raise ValueError(f"{cls.__name__} saves in format 2 only; this file has format {state.get('format')!r}")


class SequentialBackpropClassifierNetwork(BackpropClassifierNetwork[TrainableLayer]):
    """
    SequentialMultiClassBackpropClassifierNetwork over the single-output network: its output layer
    has one node. It randomizes fan-in-aware, as FanInAwareBackpropClassifierNetwork and the array
    networks, not with BackpropClassifierNetwork's bounds-width scaling. It saves and loads as the
    multiclass one does.
    """

    preset_arguments: ClassVar[tuple[str, ...] | None] = None

    def __init__(
        self,
        input_shape: InputShape,
        layers: Sequence[LayerSpec],
        update_rule: UpdateRule,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        output = layers[-1] if layers else None
        assert isinstance(output, Dense) and output.size == 1, (
            f"a single-output network's last layer is a one-node Dense; got {output!r}"
        )

        self.update_rule = update_rule

        super(BackpropClassifierNetwork, self).__init__(layers, input_shape, _bounds(input_shape, input_bounds))

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    @classmethod
    def _from_file(cls, file: NetworkFile) -> Self:
        return cls(file.input_shape, file.layers, file.update_rule, file.input_bounds)

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        # not the preset parent's envelope, which no Sequential network ever wrote
        raise ValueError(f"{cls.__name__} saves in format 2 only; this file has format {state.get('format')!r}")

    def randomize(self) -> None:
        randomize_fan_in_aware(self)
