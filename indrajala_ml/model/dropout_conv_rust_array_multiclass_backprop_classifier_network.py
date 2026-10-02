from __future__ import annotations

from collections.abc import Sequence

from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.layers.rust.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.specs.layer_specs import Dense


class DropoutConvRustArrayMultiClassBackpropClassifierNetwork(ConvRustArrayMultiClassBackpropClassifierNetwork):
    """
    DropoutConvVectorizedMultiClassBackpropClassifierNetwork on the Rust backend, with
    DropoutRustArrayLayer dense hidden layers and a plain RustArrayLayer output.
    """

    hyperparameters = ("drop_probability",)

    def __init__(
        self,
        input_height: int,
        input_width: int,
        conv_specs: Sequence[ConvSpec | PoolSpec],
        dense_layer_sizes: list[int],
        class_count: int,
        drop_probability: float,
    ) -> None:
        self.drop_probability = drop_probability
        super().__init__(input_height, input_width, conv_specs, dense_layer_sizes, class_count)
        dense_hidden = self.layers[len(self.conv_specs) : -1]
        self.hidden_layers = [layer for layer in dense_hidden if isinstance(layer, DropoutRustArrayLayer)]
        assert len(self.hidden_layers) == len(dense_hidden)  # _hidden_spec: every dense hidden layer drops out

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, dropout=self.drop_probability)
