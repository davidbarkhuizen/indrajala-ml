from __future__ import annotations

from indrajala_ml.model.layer_specs import Dense
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class ReLURustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    ReLUVectorizedMultiClassBackpropClassifierNetwork on the Rust backend: ReLURustArrayLayer hidden
    layers and a sigmoid RustArrayLayer output.
    """

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, activation="relu")
