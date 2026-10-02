from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Self

from indrajala_ml.model.layers.python.backprop_layer import BackpropLayer
from indrajala_ml.model.networks.python.backprop_network_base import BackpropNetworkBase, randomize_fan_in_aware
from indrajala_ml.model.protocols.classification import argmax_first_occurrence
from indrajala_ml.model.protocols.layer_protocols import TrainableLayer
from indrajala_ml.model.specs.bounds import validate_class_count, validate_layer_sizes


class MultiClassBackpropClassifierNetwork[LayerT: TrainableLayer = BackpropLayer](BackpropNetworkBase[LayerT]):
    """
    A one-vs-rest multiclass sibling of BackpropClassifierNetwork, from the same BackpropNode and
    BackpropLayer blocks. A separate class because classify_state returns a class index, not a
    0.0/1.0 float.

    The output layer has class_count nodes, each trained independently against a one-hot target
    (the per-node output delta needs no change); the predicted class is the most active node.
    """

    format2_shape: ClassVar[str] = "multiclass"
    preset_arguments: ClassVar[tuple[str, ...] | None] = ("layer_sizes", "dimension", "input_bounds", "class_count")

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
    ) -> None:

        validate_class_count(class_count)
        validate_layer_sizes(layer_sizes)
        self.class_count = class_count
        self.layer_sizes = layer_sizes

        super().__init__(self._dense_specs(layer_sizes, class_count), (dimension,), input_bounds)

    def _forward(self, state: tuple[float, ...]) -> list[float]:
        return self._forward_outputs(state)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return self._forward(state)

    def classify_state(self, state: tuple[float, ...]) -> int:
        return argmax_first_occurrence(self.predict_probabilities(state))

    def learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], int]]) -> None:
        self._learn_batch(learning_rate, batch)

    def _backward(self, category: int) -> None:
        self._output_deltas(category)
        self._backward_hidden_layers()

    def _output_deltas(self, category: int) -> None:
        for i, node in enumerate(self.output_layer.nodes):
            node.compute_output_delta(1.0 if i == category else 0.0)

    def randomize(self) -> None:
        # not BackpropClassifierNetwork's bounds-width scaling, which saturates every sigmoid
        # once fan-in reaches the tens (64 for 8x8 digit images)
        randomize_fan_in_aware(self)

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        # the legacy envelope: layer_sizes, dimension, input_bounds (lists in JSON) and class_count
        input_bounds = [tuple(bound) for bound in state["input_bounds"]]
        network = cls(state["layer_sizes"], state["dimension"], input_bounds, state["class_count"])
        network.restore(state["snapshot"])
        return network
