from __future__ import annotations

from indrajala_ml.model.dropout_array_layer import DropoutArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class DropoutVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The dropout sibling of VectorizedMultiClassBackpropClassifierNetwork: DropoutArrayLayer hidden
    layers, which read drop_probability (required) from the network, and a plain sigmoid output
    layer, as in DropoutBackpropClassifierNetwork.

    _set_training_mode switches the hidden layers' dropout on for the forward pass of each learn
    call (ArrayNetworkBase brackets it). self.hidden_layers is part of the public surface: tests
    check that training mode resets between calls.
    """

    hidden_layer_cls = DropoutArrayLayer
    hyperparameters = ("drop_probability",)

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, drop_probability: float) -> None:
        self.drop_probability = drop_probability
        super().__init__(layer_sizes, dimension, class_count)
        hidden = self.layers[:-1]
        self.hidden_layers = [layer for layer in hidden if isinstance(layer, DropoutArrayLayer)]
        assert len(self.hidden_layers) == len(hidden)  # hidden_layer_cls: every hidden layer drops out

    def _set_training_mode(self, training: bool) -> None:
        for layer in self.hidden_layers:
            layer.set_training_mode(training)
