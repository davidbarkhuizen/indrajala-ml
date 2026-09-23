from __future__ import annotations

from typing import Sequence

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer, fan_in_aware_random_layer
from indrajala_ml.model.bounds import validate_batch, validate_layer_sizes


class ArrayNetworkBase:
    """
    Shared machinery behind every numpy-array-backed sibling network in this codebase - plain,
    momentum, L2, Adam, ReLU, softmax, dropout, cross-entropy, both the multiclass and
    single-output shapes: layer assembly, the forward pass, learn/learn_batch, fan-in-aware
    randomize, and snapshot/restore.

    Mirrors BackpropNetworkBase's own hidden_layer_cls/output_layer_cls extension-point design
    exactly, one level up: every method here is identical across every array-based sibling in
    this codebase except for which layer class gets constructed, one constructor hyperparameter,
    and the target-array shape - a purely mechanical collapse, not a redesign of any hand-derived
    formula. Every hand-derived formula lives entirely in the ArrayLayer subclass a sibling plugs
    in via hidden_layer_cls/output_layer_cls (e.g. MomentumArrayLayer.apply_accumulated_gradient).

    What stays out of this base, in the two "shape" subclasses instead
    (VectorizedMultiClassBackpropClassifierNetwork / ArrayBackpropClassifierNetwork):
    predict_probabilities/predict_probability and classify_state (argmax vs. 0.5-threshold),
    class_count handling, and save/load (the JSON envelope and its class_count presence differ
    between the two shapes) - genuinely different concerns, not duplicated ones, matching where
    BackpropClassifierNetwork/MultiClassBackpropClassifierNetwork already draw the same line over
    BackpropNetworkBase.
    """

    # override points for a sibling whose hidden/output layers need different per-layer math
    # (e.g. MomentumArrayLayer) - every "plain" shape class leaves these as ArrayLayer, so this
    # is a pure extension point with zero behavior change for them. A hyperparameter-free
    # sibling (ReLU, softmax, cross-entropy) can set one of these as a plain class attribute; a
    # hyperparameter-bearing sibling (momentum, L2, Adam, dropout) sets it as an *instance*
    # attribute in its own __init__ (a closure capturing the hyperparameter) before calling
    # super().__init__() - the same pattern AdamBackpropClassifierNetwork already uses one layer
    # down, over BackpropNetworkBase's own hidden_layer_cls/output_layer_cls.
    hidden_layer_cls: type = ArrayLayer
    output_layer_cls: type = ArrayLayer

    def __init__(self, layer_sizes: list[int], dimension: int, output_size: int) -> None:

        validate_layer_sizes(layer_sizes)

        self.layer_sizes = layer_sizes
        self.dimension = dimension

        self.layers: list[ArrayLayer] = []
        previous_size = dimension
        for size in layer_sizes:
            self.layers.append(self.hidden_layer_cls(size, previous_size))
            previous_size = size

        self.output_layer = self.output_layer_cls(output_size, previous_size)
        self.layers.append(self.output_layer)

    def _forward(self, state: tuple[float, ...]) -> np.ndarray:
        x = np.array(state, dtype=np.float64)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _set_training_mode(self, training: bool) -> None:
        # no-op for every sibling except dropout's own override - the array-level counterpart to
        # BackpropNetworkBase._set_training_mode, and to DropoutArrayLayer.set_training_mode
        # which this hook delegates to once dropout overrides it.
        pass

    def _target_array(self, category) -> np.ndarray:
        raise NotImplementedError

    def _target_batch_array(
        self, batch: Sequence[tuple[tuple[float, ...], object]], batch_size: int
    ) -> np.ndarray:
        raise NotImplementedError

    def learn(self, learning_rate: float, state: tuple[float, ...], category) -> None:
        activations = [np.array(state, dtype=np.float64)]
        x = activations[0]
        self._set_training_mode(True)
        try:
            for layer in self.layers:
                x = layer.forward(x)
                activations.append(x)
        finally:
            self._set_training_mode(False)

        target = self._target_array(category)
        self.output_layer.compute_output_delta(target)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta(self.layers[i + 1])

        for layer, input_activation in zip(self.layers, activations):
            layer.accumulate_gradient(input_activation)
            layer.apply_accumulated_gradient(learning_rate, batch_size=1)

    def learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], object]]) -> None:
        validate_batch(batch)
        batch_size = len(batch)

        activations = [np.array([state for state, _target in batch], dtype=np.float64)]
        X = activations[0]
        self._set_training_mode(True)
        try:
            for layer in self.layers:
                X = layer.forward_batch(X)
                activations.append(X)
        finally:
            self._set_training_mode(False)

        target_batch = self._target_batch_array(batch, batch_size)
        self.output_layer.compute_output_delta_batch(target_batch)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta_batch(self.layers[i + 1])

        for layer, input_activation_batch in zip(self.layers, activations):
            layer.accumulate_gradient_batch(input_activation_batch)
            layer.apply_accumulated_gradient(learning_rate, batch_size)

    def randomize(self) -> None:
        # the same fan-in-aware scheme (limit = 1/sqrt(fan_in)) every array-based sibling in
        # this codebase would otherwise have to reimplement independently
        previous_size = self.dimension
        for layer in self.layers:
            layer.W, layer.b = fan_in_aware_random_layer(layer.size, previous_size)
            previous_size = layer.size

    def snapshot(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def restore(self, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for layer, (W, b) in zip(self.layers, snapshot):
            layer.W = np.array(W, dtype=np.float64).copy()
            layer.b = np.array(b, dtype=np.float64).copy()
