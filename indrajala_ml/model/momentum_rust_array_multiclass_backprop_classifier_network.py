from __future__ import annotations

from indrajala_ml.model.momentum_rust_array_layer import MomentumRustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class MomentumRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    MomentumVectorizedMultiClassBackpropClassifierNetwork on the Rust backend: the same network,
    with MomentumRustArrayLayer in place of MomentumArrayLayer (hidden and output layers). momentum is required, as there.
    """

    hidden_layer_cls = output_layer_cls = MomentumRustArrayLayer
    hyperparameters = ("momentum",)

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, momentum: float) -> None:
        self.momentum = momentum
        super().__init__(layer_sizes, dimension, class_count)
