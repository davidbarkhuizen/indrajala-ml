from __future__ import annotations

from indrajala_ml.model.networks.numpy.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.spec_hooks import RulePreset
from indrajala_ml.model.specs.update_rules import WeightDecay


class L2VectorizedMultiClassBackpropClassifierNetwork(RulePreset, VectorizedMultiClassBackpropClassifierNetwork):
    """
    The L2 (weight decay) sibling of VectorizedMultiClassBackpropClassifierNetwork: its optimizer
    applies the WeightDecay rule, with l2_lambda required. L2 keeps no per-parameter state, so
    snapshot()/restore() of W/b is complete.
    """

    rule = WeightDecay

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, l2_lambda: float) -> None:
        self.l2_lambda = l2_lambda
        super().__init__(layer_sizes, dimension, class_count)
