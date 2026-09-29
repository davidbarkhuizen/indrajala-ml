from __future__ import annotations

from indrajala_ml.model.layer_specs import Dense
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class ReLUVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    VectorizedMultiClassBackpropClassifierNetwork with ReLUArrayLayer hidden layers and a sigmoid
    output, as in ReLUBackpropClassifierNetwork.
    """

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, activation="relu")
