from __future__ import annotations

import math
import random
from typing import Any, Sequence

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.bounds import validate_batch, validate_input_bounds, validate_layer_sizes
from indrajala_ml.model.state_layer import StateLayer


class BackpropNetworkBase:
    """
    What BackpropClassifierNetwork and MultiClassBackpropClassifierNetwork share: layer assembly
    (input -> hidden layer(s) -> output layer), the forward pass, applying gradients, the hidden
    layers' backward pass, and snapshot/restore. The subclasses differ in the output layer's size,
    the predict_*/classify_state contract, and randomize().
    """

    # the layer classes a sibling overrides for different per-node math (e.g. SoftmaxOutputLayer,
    # ReLULayer)
    output_layer_cls: type[BackpropLayer] = BackpropLayer
    hidden_layer_cls: type[BackpropLayer] = BackpropLayer

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        output_size: int,
    ) -> None:

        validate_layer_sizes(layer_sizes)

        self.dimension = dimension

        validate_input_bounds(dimension, input_bounds)
        self.input_bounds = input_bounds

        self.input_layer = StateLayer(dimension, input_bounds)

        self.hidden_layers: list[BackpropLayer] = []
        previous_layer: StateLayer | BackpropLayer = self.input_layer
        for size in layer_sizes:
            layer = self.hidden_layer_cls(size=size, input_layer=previous_layer)
            self.hidden_layers.append(layer)
            previous_layer = layer

        self.output_layer = self.output_layer_cls(size=output_size, input_layer=previous_layer)

        # every layer with trained weights, in forward order: the backward pass and
        # snapshot/restore walk it
        self.trainable_layers: list[BackpropLayer] = self.hidden_layers + [self.output_layer]

    @classmethod
    def randomized(cls, *args, **kwargs):
        # every subclass's randomized signature is its __init__ signature; randomize() is per
        # subclass
        network = cls(*args, **kwargs)
        network.randomize()
        return network

    def update_state_layer(self, state: tuple[float, ...]) -> None:
        self.input_layer.update_state(state)

    def _forward_outputs(self, state: tuple[float, ...]) -> list[float]:
        self.update_state_layer(state)
        for layer in self.trainable_layers:
            layer.forward()
        return [node.value() for node in self.output_layer.nodes]

    def _backward_hidden_layers(self) -> None:
        for layer_index in reversed(range(len(self.hidden_layers))):
            next_layer = self.trainable_layers[layer_index + 1]
            self.hidden_layers[layer_index].compute_hidden_deltas(next_layer)

    def _set_training_mode(self, training: bool) -> None:
        # on for one training call only: the trainers classify with the same network between
        # learn steps. A no-op except for training-aware layers like DropoutLayer.
        for layer in self.trainable_layers:
            layer.set_training_mode(training)

    def _learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], Any]]) -> None:
        # forward, backward and accumulate per example, then one averaged update. A one-example
        # batch matches learn() bit for bit (tests/test_gradient_accumulation.py). The target is
        # a float or a class index; _forward/_backward abstract over which.
        validate_batch(batch)
        self._set_training_mode(True)
        try:
            for state, target in batch:
                self._forward(state)
                self._backward(target)
                self._accumulate_gradients()
            self._apply_accumulated_gradients(learning_rate, len(batch))
        finally:
            self._set_training_mode(False)

    def _apply_gradients(self, learning_rate: float) -> None:
        for layer in self.trainable_layers:
            layer.apply_gradients(learning_rate)

    def _accumulate_gradients(self) -> None:
        for layer in self.trainable_layers:
            layer.accumulate_gradients()

    def _apply_accumulated_gradients(self, learning_rate: float, batch_size: int) -> None:
        for layer in self.trainable_layers:
            layer.apply_accumulated_gradients(learning_rate, batch_size)

    def snapshot(self) -> list[list[tuple[list[float], float]]]:
        return [layer.snapshot_state() for layer in self.trainable_layers]

    def restore(self, snapshot: list[list[tuple[list[float], float]]]) -> None:
        for layer, layer_snapshot in zip(self.trainable_layers, snapshot):
            layer.restore_state(layer_snapshot)


def fan_in_aware_weights_and_bias(fan_in: int) -> tuple[list[float], float]:
    """
    fan_in weights and a bias drawn uniformly from [-limit, limit], limit = 1/sqrt(fan_in): the
    formula every fan-in-aware initialization uses (randomize_fan_in_aware,
    ConvKernel.randomize_fan_in_aware, the conv network's dense tail).
    """
    limit = 1.0 / math.sqrt(fan_in)
    weights = [random.uniform(-limit, limit) for _ in range(fan_in)]
    bias = random.uniform(-limit, limit)
    return weights, bias


def randomize_fan_in_aware(network: BackpropNetworkBase) -> None:
    """
    Fan-in-aware initialization, limit = 1/sqrt(fan_in) per layer, so a layer's weighted input sum
    doesn't saturate every sigmoid once fan-in reaches the tens or hundreds. On UCI digits: 99.5%
    training and 96.9% test accuracy. Used by MultiClassBackpropClassifierNetwork and
    FanInAwareBackpropClassifierNetwork.

    Unlike BackpropClassifierNetwork.randomize()'s bounds-width scaling (tuned for 1-2D geometric
    problems) it works at any dimension: the ensemble's 784-pixel MNIST sub-networks reach 6.6
    points higher test accuracy with it; bounds-width scaling leaves 83.5% of hidden activations
    saturated at initialization.
    """

    previous_size = network.dimension
    for layer in network.trainable_layers:
        for node in layer.nodes:
            weights, bias = fan_in_aware_weights_and_bias(previous_size)
            node.update_input_weights(weights)
            node.bias = bias
        previous_size = layer.size
