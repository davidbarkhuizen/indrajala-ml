from __future__ import annotations

from indrajala_ml.model.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class DropoutRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    The Rust-matmul-backed counterpart to DropoutVectorizedMultiClassBackpropClassifierNetwork.
    Hidden layers are
    built from DropoutRustArrayLayer (with drop_probability bound via a closure); the output
    layer stays the inherited plain RustArrayLayer (sigmoid) - the same hidden-layer-only split
    DropoutVectorizedMultiClassBackpropClassifierNetwork uses. drop_probability is a required
    constructor argument, no default, matching that class's own posture.

    _set_training_mode overrides the base class's no-op hook the same way
    DropoutVectorizedMultiClassBackpropClassifierNetwork's own override does - see that class's
    docstring for the full reasoning, unchanged here, including keeping self.hidden_layers as a
    real attribute rather than a local slice.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, drop_probability: float) -> None:
        self.drop_probability = drop_probability
        self.hidden_layer_cls = lambda size, input_size: DropoutRustArrayLayer(size, input_size, drop_probability)
        super().__init__(layer_sizes, dimension, class_count)
        self.hidden_layers = self.layers[:-1]

    def _set_training_mode(self, training: bool) -> None:
        for layer in self.hidden_layers:
            layer.set_training_mode(training)

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
        drop_probability: float,
    ) -> "DropoutRustArrayMultiClassBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, class_count, drop_probability)
        network.randomize()
        return network

    def _extra_state(self) -> dict:
        return {"drop_probability": self.drop_probability}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {"drop_probability": state["drop_probability"]}
