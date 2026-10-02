from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.specs.update_rules import Momentum


class MomentumConvMultiClassBackpropClassifierNetwork(ConvMultiClassBackpropClassifierNetwork):
    """
    A momentum sibling of ConvMultiClassBackpropClassifierNetwork: its optimizer applies the
    Momentum rule, Goyal et al. 2017's eq. (9), to every conv kernel and dense node. g is a
    kernel's accumulator, summed over positions and examples, so g / B averages examples only.
    Pool layers have no weights, so they are unchanged.

    momentum is required. A format-2 save holds it and the velocities; the legacy envelope held
    momentum only.
    """

    hyperparameters: ClassVar[tuple[str, ...]] = ("momentum",)

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
