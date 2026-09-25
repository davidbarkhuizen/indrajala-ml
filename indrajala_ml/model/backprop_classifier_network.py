from __future__ import annotations

import random
from typing import Sequence

from indrajala_ml.model.backprop_network_base import BackpropNetworkBase
from indrajala_ml.model.bounds import half_widths as _half_widths


class BackpropClassifierNetwork(BackpropNetworkBase):
    """
    A sigmoid-activation, gradient-descent-trained network of arbitrary depth
    (input -> hidden layer(s) -> a trainable single-node output layer), added alongside
    LinearClassifierNetwork rather than as a retrofit of it: AssociationNode's hard step
    function and discrete minimum-disturbance update rule are fundamentally different from
    gradient-based learning, and every hidden node here can contribute either positively or
    negatively to the output (unlike LinearClassifierNetwork's fixed weight-of-1.0-per-node
    output layer, which can only ever be a monotonically non-decreasing function of how many
    hidden nodes are active) - see demo_xor_linear_classifier_ceiling.py for what that restriction
    can't express, and demo_xor_backprop_convergence.py for this class succeeding on exactly that target.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
    ) -> None:
        super().__init__(layer_sizes, dimension, input_bounds, output_size=1)

    def _forward(self, state: tuple[float, ...]) -> float:
        return self._forward_outputs(state)[0]

    def predict_probability(self, state: tuple[float, ...]) -> float:
        return self._forward(state)

    def classify_state(self, state: tuple[float, ...]) -> float:
        return 1.0 if self.predict_probability(state) > 0.5 else 0.0

    def learn(self, learning_rate: float, state: tuple[float, ...], category: float) -> None:
        self._set_training_mode(True)
        try:
            self._forward(state)
        finally:
            self._set_training_mode(False)
        self._backward(category)
        self._apply_gradients(learning_rate)

    def learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], float]]) -> None:
        self._learn_batch(learning_rate, batch)

    def _backward(self, reference_value: float) -> None:
        self.output_layer.nodes[0].compute_output_delta(reference_value)
        self._backward_hidden_layers()

    def half_widths(self) -> list[float]:
        return _half_widths(self.input_bounds)

    def randomize(self) -> None:
        # first hidden layer: same scale-invariance rationale as
        # LinearClassifierNetwork.randomize() - each weight's range scales inversely with its
        # own input dimension's half-width, so a random node has a similar chance of splitting
        # the input space regardless of input_bounds' scale. Every later trainable layer's
        # inputs are already sigmoid-normalised to (0, 1), so a fixed small range suffices.
        # Independent random draws per node (not a shared default) are required here, unlike
        # AssociationNode's identical default weights being harmless - identical starting
        # weights across nodes in a BackpropLayer would receive identical gradients forever
        # and the layer would collapse to one effective unit.
        half_widths = self.half_widths()
        for node in self.hidden_layers[0].nodes:
            node.update_input_weights(
                [random.uniform(-2.0 / half_width, 2.0 / half_width) for half_width in half_widths]
            )
            node.bias = random.uniform(-1.0, 1.0)

        for layer in self.trainable_layers[1:]:
            for node in layer.nodes:
                node.update_input_weights([random.uniform(-1.0, 1.0) for _ in node.input_nodes])
                node.bias = random.uniform(-1.0, 1.0)
