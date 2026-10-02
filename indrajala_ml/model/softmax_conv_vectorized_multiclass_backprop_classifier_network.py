from __future__ import annotations

from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense


class SoftmaxConvVectorizedMultiClassBackpropClassifierNetwork(ConvVectorizedMultiClassBackpropClassifierNetwork):
    """
    ConvVectorizedMultiClassBackpropClassifierNetwork with a SoftmaxArrayLayer output softmax with
    cross-entropy loss, as in SoftmaxMultiClassBackpropClassifierNetwork, so predict_probabilities
    sums to 1.0. The numpy counterpart of SoftmaxConvMultiClassBackpropClassifierNetwork.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, activation="softmax", loss="cross_entropy")
