from __future__ import annotations

from indrajala_ml.model.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class DropoutRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    DropoutVectorizedMultiClassBackpropClassifierNetwork on the Rust backend, with
    DropoutRustArrayLayer hidden layers and a plain RustArrayLayer output.
    """

    hidden_layer_cls = DropoutRustArrayLayer
    hyperparameters = ("drop_probability",)

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, drop_probability: float) -> None:
        self.drop_probability = drop_probability
        super().__init__(layer_sizes, dimension, class_count)
        hidden = self.layers[:-1]
        self.hidden_layers = [layer for layer in hidden if isinstance(layer, DropoutRustArrayLayer)]
        assert len(self.hidden_layers) == len(hidden)  # hidden_layer_cls: every hidden layer drops out

    def _set_training_mode(self, training: bool) -> None:
        for layer in self.hidden_layers:
            layer.set_training_mode(training)
