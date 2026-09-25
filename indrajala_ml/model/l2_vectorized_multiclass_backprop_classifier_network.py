from __future__ import annotations

from indrajala_ml.model.l2_array_layer import L2ArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class L2VectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The L2 (weight decay) sibling of VectorizedMultiClassBackpropClassifierNetwork: hidden and
    output layers are L2ArrayLayers, which read l2_lambda (required) from the network. L2 keeps no
    per-parameter state, so snapshot()/restore() of W/b is complete.
    """

    hidden_layer_cls = output_layer_cls = L2ArrayLayer
    hyperparameters = ("l2_lambda",)

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, l2_lambda: float) -> None:
        self.l2_lambda = l2_lambda
        super().__init__(layer_sizes, dimension, class_count)
