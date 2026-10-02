from __future__ import annotations

from collections.abc import Sequence

from indrajala_ml.model.networks.rust.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.specs.update_rules import Momentum


class MomentumConvRustArrayMultiClassBackpropClassifierNetwork(ConvRustArrayMultiClassBackpropClassifierNetwork):
    """
    MomentumConvVectorizedMultiClassBackpropClassifierNetwork on the Rust backend: the optimizer's
    Momentum rule for the conv, dense and output layers.
    """

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

    def _update_rule(self) -> Momentum:
        return Momentum(self.momentum)
