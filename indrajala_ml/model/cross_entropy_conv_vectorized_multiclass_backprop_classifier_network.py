from __future__ import annotations

from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.layer_specs import Dense


class CrossEntropyConvVectorizedMultiClassBackpropClassifierNetwork(ConvVectorizedMultiClassBackpropClassifierNetwork):
    """
    ConvVectorizedMultiClassBackpropClassifierNetwork with a CrossEntropyArrayLayer output: an
    independent cross-entropy delta at each of the class_count sigmoid outputs, as in
    CrossEntropyMultiClassBackpropClassifierNetwork. predict_probabilities doesn't sum to 1.0;
    classify_state is the argmax. The numpy counterpart of
    CrossEntropyConvMultiClassBackpropClassifierNetwork.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, loss="cross_entropy")
