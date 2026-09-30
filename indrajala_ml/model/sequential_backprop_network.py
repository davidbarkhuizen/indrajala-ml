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
from typing import Self

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.backprop_network_base import randomize_fan_in_aware
from indrajala_ml.model.bounds import validate_class_count
from indrajala_ml.model.layer_protocols import TrainableLayer
from indrajala_ml.model.layer_specs import Dense, InputShape, LayerSpec, sequential_save_not_yet
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.update_rules import UpdateRule


def _unit_bounds(input_shape: InputShape) -> list[tuple[float, float]]:
    # the input layer's bounds, which only BackpropClassifierNetwork's bounds-width randomize and
    # the legacy save envelopes read, neither of which a sequential network uses
    return [(0.0, 1.0)] * math.prod(input_shape)


class SequentialMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork[TrainableLayer]):
    """
    A multiclass network of any accepted layer specs (layer_specs.py) and update rule, with the
    multiclass network's argmax classification, one-hot targets and fan-in-aware randomize.
    class_count is the output layer's size. It has no save/load until format 2 (stage 5 of
    docs/composable-layers-workplan.md).
    """

    def __init__(self, input_shape: InputShape, layers: Sequence[LayerSpec], update_rule: UpdateRule) -> None:
        output = layers[-1] if layers else None
        assert isinstance(output, Dense), f"the last layer must be the output layer, a Dense; got {output!r}"
        validate_class_count(output.size)

        self.class_count = output.size
        self.input_shape = input_shape
        self.layer_specs = list(layers)
        self.update_rule = update_rule

        # past the multiclass network's __init__, whose flat layer_sizes can't describe these layers,
        # to BackpropNetworkBase's
        super(MultiClassBackpropClassifierNetwork, self).__init__(
            self.layer_specs, input_shape, _unit_bounds(input_shape)
        )

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    def save(self, path: str) -> None:
        sequential_save_not_yet(self)

    @classmethod
    def load(cls, path: str) -> Self:
        sequential_save_not_yet(cls)


class SequentialBackpropClassifierNetwork(BackpropClassifierNetwork[TrainableLayer]):
    """
    SequentialMultiClassBackpropClassifierNetwork over the single-output network: its output layer
    has one node. It randomizes fan-in-aware, as FanInAwareBackpropClassifierNetwork and the array
    networks, not with BackpropClassifierNetwork's bounds-width scaling. Like every pure-Python
    single-output network, it doesn't save.
    """

    def __init__(self, input_shape: InputShape, layers: Sequence[LayerSpec], update_rule: UpdateRule) -> None:
        output = layers[-1] if layers else None
        assert isinstance(output, Dense) and output.size == 1, (
            f"a single-output network's last layer is a one-node Dense; got {output!r}"
        )

        self.input_shape = input_shape
        self.layer_specs = list(layers)
        self.update_rule = update_rule

        super(BackpropClassifierNetwork, self).__init__(self.layer_specs, input_shape, _unit_bounds(input_shape))

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    def randomize(self) -> None:
        randomize_fan_in_aware(self)
