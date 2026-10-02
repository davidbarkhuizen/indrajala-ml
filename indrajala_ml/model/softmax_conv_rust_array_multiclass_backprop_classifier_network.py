from __future__ import annotations

from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.layer_specs import Dense


class SoftmaxConvRustArrayMultiClassBackpropClassifierNetwork(ConvRustArrayMultiClassBackpropClassifierNetwork):
    """
    SoftmaxConvVectorizedMultiClassBackpropClassifierNetwork on the Rust backend, with a
    SoftmaxRustArrayLayer output.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, activation="softmax", loss="cross_entropy")
