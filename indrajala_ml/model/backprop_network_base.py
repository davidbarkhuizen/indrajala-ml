from __future__ import annotations

import math
import random
from typing import Any, Sequence

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.bounds import validate_batch, validate_input_bounds, validate_layer_sizes
from indrajala_ml.model.state_layer import StateLayer


class BackpropNetworkBase:
    """
    Shared machinery behind BackpropClassifierNetwork and MultiClassBackpropClassifierNetwork:
    layer assembly (input -> hidden layer(s) -> output layer), the forward pass, gradient
    application, the hidden-layer half of backprop, and snapshot/restore. The two subclasses
    differ only in output-layer size/shape (a single node vs class_count nodes), the resulting
    predict_*/classify_state contract, and randomize()'s initialization scheme (see each
    subclass's own docstring) - genuinely different concerns, not duplicated ones, so they stay
    out of this base.
    """

    # override point for a subclass whose output layer needs different per-node activation
    # semantics (e.g. a softmax multi-class sibling's SoftmaxOutputLayer) - every existing
    # subclass leaves this as plain BackpropLayer, so this is a pure extension point with zero
    # behavior change for them
    output_layer_cls: type[BackpropLayer] = BackpropLayer

    # same override point, for the hidden layers instead (e.g. a ReLU sibling's ReLULayer) -
    # every existing subclass leaves this as plain BackpropLayer too
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

        # drives both the backward pass and snapshot/restore uniformly - every layer whose
        # weights/bias are actually trained, in forward order
        self.trainable_layers: list[BackpropLayer] = self.hidden_layers + [self.output_layer]

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
        # call-scoped, not lifecycle-scoped: train.py's own training loops call
        # classify_state()/predict_probability() on the same, mid-training student between
        # (not just after) learn()/learn_batch() steps, so this must be toggled on for the
        # duration of one training call and off again immediately after - never left on. A
        # no-op for every layer except a training-aware sibling like DropoutLayer.
        for layer in self.trainable_layers:
            layer.set_training_mode(training)

    def _learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], Any]]) -> None:
        # the batch-shaped analogue of learn(): forward+backward+accumulate once per example,
        # then a single averaged weight update - batch_size=1 (a one-element batch) is required
        # to match learn()'s own result exactly, since apply_accumulated_gradient's batch_size=1
        # case is already proven bit-identical to apply_gradient (see
        # tests/test_gradient_accumulation.py). Shared by BackpropClassifierNetwork's and
        # MultiClassBackpropClassifierNetwork's own learn_batch() - identical shape, differing
        # only in what a "target" is (a float reference value vs. an int category), which
        # _forward/_backward already abstract over.
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
    Draws fan_in weights plus one bias uniformly from [-limit, limit], limit = 1/sqrt(fan_in) -
    the shared core of every fan-in-aware initialization scheme in this codebase: dense layers
    below (randomize_fan_in_aware), ConvKernel.randomize_fan_in_aware (conv_kernel.py), and
    ConvMultiClassBackpropClassifierNetwork.randomize's own dense tail
    (conv_multiclass_backprop_classifier_network.py) - one formula, one place to change it.
    """
    limit = 1.0 / math.sqrt(fan_in)
    weights = [random.uniform(-limit, limit) for _ in range(fan_in)]
    bias = random.uniform(-limit, limit)
    return weights, bias


def randomize_fan_in_aware(network: BackpropNetworkBase) -> None:
    """
    Fan-in-aware weight/bias initialization (limit = 1/sqrt(fan_in) per layer) - each weight
    drawn uniformly from [-limit, limit], scaled down as fan-in grows, so a layer's weighted
    input sum doesn't blow up (guaranteeing sigmoid saturation at every node) once fan-in
    reaches the tens or hundreds. Validated against the real bundled UCI digits dataset (99.5%
    training accuracy, 96.9% test accuracy). Shared by
    MultiClassBackpropClassifierNetwork.randomize() and
    FanInAwareBackpropClassifierNetwork.randomize(), so both classes use one implementation
    instead of two copies of the same formula.

    Unlike BackpropClassifierNetwork.randomize()'s per-dimension-bounds-width scaling (tuned for
    1-2D geometric problems - see that method's own docstring), this scheme is dimension-generic:
    at real scale, EnsembleBackpropClassifierNetwork's 784-dimension MNIST sub-networks reach
    +6.6 points higher test accuracy with fan-in-aware init than with per-dimension-bounds-width
    scaling, which leaves 83.5% of hidden activations already saturated at initialization.
    """

    previous_size = network.dimension
    for layer in network.trainable_layers:
        for node in layer.nodes:
            weights, bias = fan_in_aware_weights_and_bias(previous_size)
            node.update_input_weights(weights)
            node.bias = bias
        previous_size = layer.size
