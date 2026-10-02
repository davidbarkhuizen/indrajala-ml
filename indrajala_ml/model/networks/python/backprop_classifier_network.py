from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from indrajala_ml.model.layers.python.backprop_layer import BackpropLayer
from indrajala_ml.model.networks.python.backprop_network_base import BackpropNetworkBase, as_dense_layers
from indrajala_ml.model.protocols.layer_protocols import TrainableLayer
from indrajala_ml.model.specs.bounds import half_widths as _half_widths
from indrajala_ml.model.specs.bounds import validate_layer_sizes


class BackpropClassifierNetwork[LayerT: TrainableLayer = BackpropLayer](BackpropNetworkBase[LayerT]):
    """
    A sigmoid network of any depth trained by gradient descent: input -> hidden layer(s) -> one
    trainable output node. Separate from LinearClassifierNetwork, not a retrofit: AssociationNode's
    hard step and minimum-disturbance rule aren't gradient-based, and its fixed weight-1.0 output
    layer can only be a non-decreasing function of how many hidden nodes fire, while here each
    hidden node can push the output either way. demo_xor_linear_classifier_ceiling.py shows what the
    restriction can't express, and demo_xor_backprop_convergence.py this class learning it.
    """

    format2_shape: ClassVar[str] = "single_output"
    preset_arguments: ClassVar[tuple[str, ...] | None] = ("layer_sizes", "dimension", "input_bounds")

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
    ) -> None:
        validate_layer_sizes(layer_sizes)
        self.layer_sizes = layer_sizes
        super().__init__(self._dense_specs(layer_sizes, 1), (dimension,), input_bounds)

    def _forward(self, state: tuple[float, ...]) -> float:
        return self._forward_outputs(state)[0]

    def predict_probability(self, state: tuple[float, ...]) -> float:
        return self._forward(state)

    def classify_state(self, state: tuple[float, ...]) -> float:
        return 1.0 if self.predict_probability(state) > 0.5 else 0.0

    def learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], float]]) -> None:
        self._learn_batch(learning_rate, batch)

    def _backward(self, reference_value: float) -> None:
        self._output_deltas(reference_value)
        self._backward_hidden_layers()

    def _output_deltas(self, reference_value: float) -> None:
        self.output_layer.nodes[0].compute_output_delta(reference_value)

    def half_widths(self) -> list[float]:
        return _half_widths(self.input_bounds)

    def randomize(self) -> None:
        # first hidden layer: as LinearClassifierNetwork.randomize(), each weight's range scales
        # inversely with its input dimension's half-width, so a random node has a similar chance
        # of splitting the input space whatever input_bounds' scale. Later layers' inputs are
        # sigmoid outputs in (0, 1), so a fixed range suffices. Every node draws independently:
        # identical starting weights would get identical gradients forever, collapsing the
        # layer to one effective unit.
        first, *later = as_dense_layers(self.trainable_layers)
        half_widths = self.half_widths()
        for node in first.nodes:
            node.update_input_weights(
                [self.rng.uniform(-2.0 / half_width, 2.0 / half_width) for half_width in half_widths]
            )
            node.bias = self.rng.uniform(-1.0, 1.0)

        for layer in later:
            for node in layer.nodes:
                node.update_input_weights([self.rng.uniform(-1.0, 1.0) for _ in node.input_nodes])
                node.bias = self.rng.uniform(-1.0, 1.0)
