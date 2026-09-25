from __future__ import annotations

from collections.abc import Sequence

from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.momentum_conv_rust_array_layer import MomentumConvRustArrayLayer
from indrajala_ml.model.momentum_rust_array_layer import MomentumRustArrayLayer


class MomentumConvRustArrayMultiClassBackpropClassifierNetwork(ConvRustArrayMultiClassBackpropClassifierNetwork):
    """
    MomentumConvVectorizedMultiClassBackpropClassifierNetwork on the Rust backend, with
    MomentumConvRustArrayLayer for the conv layers and MomentumRustArrayLayer for the dense and
    output layers.
    """

    conv_layer_cls = MomentumConvRustArrayLayer
    hidden_layer_cls = output_layer_cls = MomentumRustArrayLayer
    hyperparameters = ("momentum",)

    def __init__(
        self,
        input_height: int,
        input_width: int,
        conv_specs: Sequence[ConvSpec | PoolSpec],
        dense_layer_sizes: list[int],
        class_count: int,
        momentum: float,
    ) -> None:
        self.momentum = momentum
        super().__init__(input_height, input_width, conv_specs, dense_layer_sizes, class_count)
