from __future__ import annotations

from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.specs.layer_specs import Dense


class ReLUMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    MultiClassBackpropClassifierNetwork with ReLULayer hidden layers and a sigmoid output, as in
    ReLUBackpropClassifierNetwork. The pure-Python sibling of
    ReLUVectorizedMultiClassBackpropClassifierNetwork and
    ReLURustArrayMultiClassBackpropClassifierNetwork.
    """

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, activation="relu")
