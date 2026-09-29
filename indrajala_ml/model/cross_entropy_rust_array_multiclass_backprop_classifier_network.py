from __future__ import annotations

from indrajala_ml.model.layer_specs import Dense
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class CrossEntropyRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    CrossEntropyVectorizedMultiClassBackpropClassifierNetwork on the Rust backend, with a
    CrossEntropyRustArrayLayer output.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, loss="cross_entropy")
