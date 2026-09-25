from __future__ import annotations

from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_rust_array_layer import SoftmaxRustArrayLayer


class SoftmaxRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    SoftmaxVectorizedMultiClassBackpropClassifierNetwork on the Rust backend: the same network,
    with SoftmaxRustArrayLayer in place of SoftmaxArrayLayer. Hidden layers stay plain
    RustArrayLayer (sigmoid); only the output layer is softmax.

    No hyperparameter and no extra constructor parameter, so nothing beyond this one
    class-attribute override is needed - __init__/randomized/save/load are all inherited
    unchanged from RustArrayMultiClassBackpropClassifierNetwork.
    """

    output_layer_cls = SoftmaxRustArrayLayer
