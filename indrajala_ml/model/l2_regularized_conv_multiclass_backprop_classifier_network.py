from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.specs.update_rules import WeightDecay


class L2RegularizedConvMultiClassBackpropClassifierNetwork(ConvMultiClassBackpropClassifierNetwork):
    """
    The L2 (weight decay) sibling of ConvMultiClassBackpropClassifierNetwork: its optimizer applies
    the WeightDecay rule, with l2_lambda required, to every conv kernel and dense node. Pool layers
    have no weights, so they are unchanged. The pure-Python sibling of
    L2ConvVectorizedMultiClassBackpropClassifierNetwork and
    L2ConvRustArrayMultiClassBackpropClassifierNetwork.
    """

    hyperparameters: ClassVar[tuple[str, ...]] = ("l2_lambda",)

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
