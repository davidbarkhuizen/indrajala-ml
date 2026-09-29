from __future__ import annotations

from collections.abc import Sequence

from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.momentum_conv_array_layer import MomentumConvArrayLayer
from indrajala_ml.model.update_rules import Momentum


class MomentumConvVectorizedMultiClassBackpropClassifierNetwork(ConvVectorizedMultiClassBackpropClassifierNetwork):
    """
    The momentum sibling of ConvVectorizedMultiClassBackpropClassifierNetwork, and the numpy
    counterpart of MomentumConvMultiClassBackpropClassifierNetwork: MomentumConvArrayLayers, which
    read momentum (required) from the network, and the optimizer's Momentum rule for the dense and
    output layers. Pool layers have no weights, so they are unchanged. momentum is saved in the
    conv envelope; the velocities are not, as for every momentum network.
    """

    conv_layer_cls = MomentumConvArrayLayer
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
