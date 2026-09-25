from __future__ import annotations

from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_rust_array_layer import SoftmaxRustArrayLayer


class SoftmaxRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    SoftmaxVectorizedMultiClassBackpropClassifierNetwork on the Rust backend, with a
    SoftmaxRustArrayLayer output.
    """

    output_layer_cls = SoftmaxRustArrayLayer
