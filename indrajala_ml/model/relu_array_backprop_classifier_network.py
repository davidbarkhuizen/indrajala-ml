from __future__ import annotations

from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.layer_specs import Dense


class ReLUArrayBackpropClassifierNetwork(ArrayBackpropClassifierNetwork):
    """
    ArrayBackpropClassifierNetwork with ReLUArrayLayer hidden layers and a sigmoid output, as in
    ReLUBackpropClassifierNetwork and ReLUVectorizedMultiClassBackpropClassifierNetwork.
    """

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, activation="relu")
