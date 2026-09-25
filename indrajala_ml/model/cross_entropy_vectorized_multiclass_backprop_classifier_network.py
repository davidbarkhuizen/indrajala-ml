from __future__ import annotations

from indrajala_ml.model.cross_entropy_array_layer import CrossEntropyArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class CrossEntropyVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    VectorizedMultiClassBackpropClassifierNetwork with a CrossEntropyArrayLayer output: an
    independent cross-entropy delta at each of the class_count sigmoid outputs, one-vs-rest rather
    than softmax's joint normalization. predict_probabilities doesn't sum to 1.0; classify_state is
    the argmax.
    """

    output_layer_cls = CrossEntropyArrayLayer
