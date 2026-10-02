from __future__ import annotations

from indrajala_ml.model.networks.python.conv_multiclass_backprop_classifier_network import (
    ConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense


class SoftmaxConvMultiClassBackpropClassifierNetwork(ConvMultiClassBackpropClassifierNetwork):
    """
    ConvMultiClassBackpropClassifierNetwork with a SoftmaxOutputLayer output softmax with cross-
    entropy loss, as in SoftmaxMultiClassBackpropClassifierNetwork, so predict_probabilities sums to
    1.0. The pure-Python sibling of SoftmaxConvVectorizedMultiClassBackpropClassifierNetwork and
    SoftmaxConvRustArrayMultiClassBackpropClassifierNetwork.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, activation="softmax", loss="cross_entropy")
