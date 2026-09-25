from __future__ import annotations

from indrajala_ml.model.l2_rust_array_layer import L2RustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class L2RustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    L2VectorizedMultiClassBackpropClassifierNetwork on the Rust backend, with L2RustArrayLayer for
    the hidden and output layers.
    """

    hidden_layer_cls = output_layer_cls = L2RustArrayLayer
    hyperparameters = ("l2_lambda",)

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, l2_lambda: float) -> None:
        self.l2_lambda = l2_lambda
        super().__init__(layer_sizes, dimension, class_count)
