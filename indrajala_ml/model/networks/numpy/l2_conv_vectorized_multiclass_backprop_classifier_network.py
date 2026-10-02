from __future__ import annotations

from collections.abc import Sequence

from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.networks.numpy.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.update_rules import WeightDecay


class L2ConvVectorizedMultiClassBackpropClassifierNetwork(ConvVectorizedMultiClassBackpropClassifierNetwork):
    """
    The L2 (weight decay) sibling of ConvVectorizedMultiClassBackpropClassifierNetwork, and the
    numpy counterpart of L2RegularizedConvMultiClassBackpropClassifierNetwork: the optimizer's
    WeightDecay rule for the conv, dense and output layers (l2_lambda is required). Pool layers have
    no weights, so they are unchanged.
    """

    hyperparameters = ("l2_lambda",)

    def __init__(
        self,
        input_height: int,
        input_width: int,
        conv_specs: Sequence[ConvSpec | PoolSpec],
        dense_layer_sizes: list[int],
        class_count: int,
        l2_lambda: float,
    ) -> None:
        self.l2_lambda = l2_lambda
        super().__init__(input_height, input_width, conv_specs, dense_layer_sizes, class_count)

    def _update_rule(self) -> WeightDecay:
        return WeightDecay(self.l2_lambda)
