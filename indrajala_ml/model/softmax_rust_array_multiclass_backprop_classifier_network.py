from __future__ import annotations

from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense


class SoftmaxRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    SoftmaxVectorizedMultiClassBackpropClassifierNetwork on the Rust backend, with a
    SoftmaxRustArrayLayer output.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, activation="softmax", loss="cross_entropy")
