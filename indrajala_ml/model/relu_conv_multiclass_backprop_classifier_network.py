from __future__ import annotations

from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.layer_specs import Dense


class ReLUConvMultiClassBackpropClassifierNetwork(ConvMultiClassBackpropClassifierNetwork):
    """
    ConvMultiClassBackpropClassifierNetwork with ReLULayer dense hidden layers after the conv front
    end, whose conv layers are already ReLU, and a sigmoid output, as in
    ReLUMultiClassBackpropClassifierNetwork. The pure-Python sibling of
    ReLUConvVectorizedMultiClassBackpropClassifierNetwork and
    ReLUConvRustArrayMultiClassBackpropClassifierNetwork.
    """

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, activation="relu")
