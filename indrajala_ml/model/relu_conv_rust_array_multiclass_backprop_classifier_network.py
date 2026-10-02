from __future__ import annotations

from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense


class ReLUConvRustArrayMultiClassBackpropClassifierNetwork(ConvRustArrayMultiClassBackpropClassifierNetwork):
    """
    ReLUConvVectorizedMultiClassBackpropClassifierNetwork on the Rust backend, with
    ReLURustArrayLayer dense hidden layers.
    """

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, activation="relu")
