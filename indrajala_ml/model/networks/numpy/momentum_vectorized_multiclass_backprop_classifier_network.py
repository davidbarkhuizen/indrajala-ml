from __future__ import annotations

from indrajala_ml.model.networks.numpy.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.spec_hooks import RulePreset
from indrajala_ml.model.specs.update_rules import Momentum


class MomentumVectorizedMultiClassBackpropClassifierNetwork(RulePreset, VectorizedMultiClassBackpropClassifierNetwork):
    """
    The momentum sibling of VectorizedMultiClassBackpropClassifierNetwork: its optimizer applies
    the Momentum rule, with momentum required. snapshot()/restore() cover only W/b, not the
    optimizer's velocities.
    """

    rule = Momentum

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, momentum: float) -> None:
        self.momentum = momentum
        super().__init__(layer_sizes, dimension, class_count)
