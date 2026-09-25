from __future__ import annotations

from indrajala_ml.model.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class DropoutRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    DropoutVectorizedMultiClassBackpropClassifierNetwork on the Rust backend: the same network,
    with DropoutRustArrayLayer in place of DropoutArrayLayer (hidden layers only,
    drop_probability bound via a closure); the output layer stays a plain RustArrayLayer
    (sigmoid). drop_probability is required, as there.

    _set_training_mode and self.hidden_layers are the numpy class's, repeated; see its docstring
    for why.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, drop_probability: float) -> None:
        self.drop_probability = drop_probability
        self.hidden_layer_cls = lambda size, input_size: DropoutRustArrayLayer(size, input_size, drop_probability)
        super().__init__(layer_sizes, dimension, class_count)
        self.hidden_layers = self.layers[:-1]

    def _set_training_mode(self, training: bool) -> None:
        for layer in self.hidden_layers:
            layer.set_training_mode(training)

    def _extra_state(self) -> dict:
        return {"drop_probability": self.drop_probability}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {"drop_probability": state["drop_probability"]}
