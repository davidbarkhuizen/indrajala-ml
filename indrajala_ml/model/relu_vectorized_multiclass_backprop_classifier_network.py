from __future__ import annotations

from indrajala_ml.model.relu_array_layer import ReLUArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class ReLUVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    VectorizedMultiClassBackpropClassifierNetwork with ReLUArrayLayer hidden layers and a sigmoid
    output, as in ReLUBackpropClassifierNetwork.
    """

    hidden_layer_cls = ReLUArrayLayer
