from __future__ import annotations

from indrajala_ml.model.adam_array_layer import AdamArrayLayer
from indrajala_ml.model.adam_backprop_classifier_network import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class AdamVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The Adam sibling of VectorizedMultiClassBackpropClassifierNetwork: hidden and output layers are
    AdamArrayLayers, which read beta1/beta2/epsilon from the network. They default to Kingma & Ba's
    published values, as in AdamBackpropClassifierNetwork.

    snapshot()/restore() cover only W/b, not Adam's m/v/t, as every network's do; resuming training
    with m/v/t intact would need an extended envelope.
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
