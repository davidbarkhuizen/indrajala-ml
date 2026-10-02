from __future__ import annotations

from indrajala_ml.model.networks.numpy.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.specs.layer_specs import Dense


class CrossEntropyArrayBackpropClassifierNetwork(ArrayBackpropClassifierNetwork):
    """
    ArrayBackpropClassifierNetwork with a CrossEntropyArrayLayer output: the numpy form of
    BinaryCrossEntropyBackpropClassifierNetwork.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, loss="cross_entropy")
