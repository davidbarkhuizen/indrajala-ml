from __future__ import annotations

from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.cross_entropy_array_layer import CrossEntropyArrayLayer


class CrossEntropyArrayBackpropClassifierNetwork(ArrayBackpropClassifierNetwork):
    """
    The single-output numpy-array-backed sibling of BinaryCrossEntropyBackpropClassifierNetwork.
    Structurally identical to
    ArrayBackpropClassifierNetwork, except its output layer is a CrossEntropyArrayLayer instead
    of a plain ArrayLayer - the array-level analogue of
    BinaryCrossEntropyBackpropClassifierNetwork's own output_layer_cls-only override, applied to
    the single-output array line instead of BackpropClassifierNetwork.

    No hyperparameter and no extra constructor parameter, so nothing beyond this one
    class-attribute override is needed - __init__/randomized/save/load are all inherited
    unchanged from ArrayBackpropClassifierNetwork.
    """

    output_layer_cls = CrossEntropyArrayLayer
