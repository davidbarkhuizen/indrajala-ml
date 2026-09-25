from __future__ import annotations

from indrajala_ml.model.momentum_array_layer import MomentumArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class MomentumVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The momentum sibling of VectorizedMultiClassBackpropClassifierNetwork: hidden and output layers
    are MomentumArrayLayers, which read momentum (required) from the network. snapshot()/restore()
    cover only W/b, not the layers' velocities.
    """

    hidden_layer_cls = output_layer_cls = MomentumArrayLayer
    hyperparameters = ("momentum",)

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, momentum: float) -> None:
        self.momentum = momentum
        super().__init__(layer_sizes, dimension, class_count)
