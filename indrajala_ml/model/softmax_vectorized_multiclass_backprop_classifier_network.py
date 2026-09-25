from __future__ import annotations

from indrajala_ml.model.softmax_array_layer import SoftmaxArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class SoftmaxVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    VectorizedMultiClassBackpropClassifierNetwork with a SoftmaxArrayLayer output. The layer's
    forward normalizes jointly, so classify_state and predict_probabilities need no override.
    """

    output_layer_cls = SoftmaxArrayLayer
