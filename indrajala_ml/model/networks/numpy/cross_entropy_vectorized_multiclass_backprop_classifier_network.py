from __future__ import annotations

from indrajala_ml.model.networks.numpy.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense


class CrossEntropyVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    VectorizedMultiClassBackpropClassifierNetwork with a CrossEntropyArrayLayer output: an
    independent cross-entropy delta at each of the class_count sigmoid outputs, one-vs-rest rather
    than softmax's joint normalization. predict_probabilities doesn't sum to 1.0; classify_state is
    the argmax.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, loss="cross_entropy")
