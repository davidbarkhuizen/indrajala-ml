from __future__ import annotations

from indrajala_ml.model.update_rules import Momentum
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class MomentumVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The momentum sibling of VectorizedMultiClassBackpropClassifierNetwork: its optimizer applies
    the Momentum rule, with momentum required. snapshot()/restore() cover only W/b, not the
    optimizer's velocities.
    """

    hyperparameters = ("momentum",)

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, momentum: float) -> None:
        self.momentum = momentum
        super().__init__(layer_sizes, dimension, class_count)

    def _update_rule(self) -> Momentum:
        return Momentum(self.momentum)
