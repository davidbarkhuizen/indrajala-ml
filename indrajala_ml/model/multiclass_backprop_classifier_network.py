from __future__ import annotations

from typing import Sequence

from indrajala_ml.model.backprop_network_base import BackpropNetworkBase, randomize_fan_in_aware
from indrajala_ml.model.bounds import validate_class_count
from indrajala_ml.model.classification import argmax_first_occurrence
from indrajala_ml.model.model_io import load_model_json, save_model_json


class MultiClassBackpropClassifierNetwork(BackpropNetworkBase):
    """
    A one-vs-rest multi-class sibling of BackpropClassifierNetwork, built entirely on the same
    BackpropNode/BackpropLayer building blocks - a new class, not a retrofit, because
    classify_state()'s return type/contract changes (a class index, not a 0.0/1.0 float), and
    every existing backprop demo depends on the binary float contract.

    The output layer has class_count nodes instead of one; each is trained independently
    against a one-hot target (BackpropNode.compute_output_delta needs no changes for this - it
    was already a per-node, sibling-independent computation). At inference, the predicted class
    is whichever output node has the highest activation.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
    ) -> None:

        validate_class_count(class_count)
        self.class_count = class_count

        super().__init__(layer_sizes, dimension, input_bounds, output_size=class_count)

    def _forward(self, state: tuple[float, ...]) -> list[float]:
        return self._forward_outputs(state)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return self._forward(state)

    def classify_state(self, state: tuple[float, ...]) -> int:
        return argmax_first_occurrence(self.predict_probabilities(state))

    def learn(self, learning_rate: float, state: tuple[float, ...], category: int) -> None:
        self._forward(state)
        self._backward(category)
        self._apply_gradients(learning_rate)

    def learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], int]]) -> None:
        self._learn_batch(learning_rate, batch)

    def _backward(self, category: int) -> None:
        for i, node in enumerate(self.output_layer.nodes):
            node.compute_output_delta(1.0 if i == category else 0.0)
        self._backward_hidden_layers()

    def randomize(self) -> None:
        # fan-in-aware initialization, unlike BackpropClassifierNetwork.randomize()'s
        # per-dimension-bounds-width scaling (tuned for 1-2D geometric problems) - that scaling
        # produces exploding pre-activation sums (guaranteed sigmoid saturation) once fan-in
        # reaches the tens or hundreds, as it does here (dimension=64 for 8x8 digit images). See
        # randomize_fan_in_aware's own docstring for the scheme itself and its validation.
        randomize_fan_in_aware(self)

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
    ) -> "MultiClassBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, input_bounds, class_count)
        network.randomize()
        return network

    def save(self, path: str) -> None:
        save_model_json(
            path,
            layer_sizes=[layer.size for layer in self.hidden_layers],
            dimension=self.dimension,
            input_bounds=self.input_bounds,
            class_count=self.class_count,
            snapshot=self.snapshot(),
        )

    @classmethod
    def load(cls, path: str) -> "MultiClassBackpropClassifierNetwork":
        state = load_model_json(path)
        network = cls(state["layer_sizes"], state["dimension"], state["input_bounds"], state["class_count"])
        network.restore(state["snapshot"])
        return network
