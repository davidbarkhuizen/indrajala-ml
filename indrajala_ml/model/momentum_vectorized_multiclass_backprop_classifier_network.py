from __future__ import annotations

from indrajala_ml.model.momentum_array_layer import MomentumArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class MomentumVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The momentum sibling of VectorizedMultiClassBackpropClassifierNetwork.

    momentum is a required constructor parameter, no default - the same posture
    MomentumBackpropClassifierNetwork's per-node counterpart already takes. Both hidden layers
    and the output layer are built from MomentumArrayLayer, which reads momentum from the
    network (ArrayNetworkBase._new_layer) - mirroring MomentumBackpropClassifierNetwork's own
    hidden_layer_cls == output_layer_cls choice, and the same pattern
    AdamBackpropClassifierNetwork already uses one layer down over BackpropNetworkBase.

    snapshot()/restore() intentionally cover only W/b, matching the base array-backed sibling's
    own contract unchanged - the same posture AdamVectorizedMultiClassBackpropClassifierNetwork
    already has (see that class's own docstring for the full reasoning), not a new gap this
    class introduces.
    """

    hidden_layer_cls = output_layer_cls = MomentumArrayLayer
    hyperparameters = ("momentum",)

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, momentum: float) -> None:
        self.momentum = momentum
        super().__init__(layer_sizes, dimension, class_count)
