from __future__ import annotations

from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.cross_entropy_array_layer import CrossEntropyArrayLayer


class CrossEntropyArrayBackpropClassifierNetwork(ArrayBackpropClassifierNetwork):
    """
    ArrayBackpropClassifierNetwork with a CrossEntropyArrayLayer output: the numpy form of
    BinaryCrossEntropyBackpropClassifierNetwork.
    """

    output_layer_cls = CrossEntropyArrayLayer
