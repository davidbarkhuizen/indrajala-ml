from __future__ import annotations

from indrajala_ml.model.adam_array_layer import AdamArrayLayer
from indrajala_ml.model.adam_backprop_classifier_network import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class AdamVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The Adam-optimized sibling of VectorizedMultiClassBackpropClassifierNetwork. Both hidden
    layers and the output layer are built from AdamArrayLayer, which reads
    beta1/beta2/epsilon from the network (ArrayNetworkBase._new_layer) - the same pattern
    momentum/L2's own array siblings use.

    beta1/beta2/epsilon default to Kingma & Ba's own published values, matching
    AdamBackpropClassifierNetwork's per-node counterpart.

    snapshot()/restore() intentionally cover only W/b, matching the base array-backed sibling's
    own contract unchanged (BackpropNetworkBase.snapshot/restore likewise only ever captured
    weights/bias, never a momentum/Adam node's own velocity/m/v/t), not a new gap this class
    introduces. A resumed-training scenario that needs m/v/t preserved across a snapshot/restore
    round trip would need its own extended envelope; no measurement in this codebase has used one
    so far.
    """

    hidden_layer_cls = output_layer_cls = AdamArrayLayer
    hyperparameters = ("beta1", "beta2", "epsilon")

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super().__init__(layer_sizes, dimension, class_count)
