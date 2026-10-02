from __future__ import annotations

from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.layer_specs import Dense


class ReLUConvVectorizedMultiClassBackpropClassifierNetwork(ConvVectorizedMultiClassBackpropClassifierNetwork):
    """
    ConvVectorizedMultiClassBackpropClassifierNetwork with ReLUArrayLayer dense hidden layers after
    the conv front end, whose conv layers are already ReLU, and a sigmoid output, as in
    ReLUMultiClassBackpropClassifierNetwork. The numpy counterpart of
    ReLUConvMultiClassBackpropClassifierNetwork.
    """

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, activation="relu")
