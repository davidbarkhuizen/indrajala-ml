from __future__ import annotations

from indrajala_ml.model.softmax_array_layer import SoftmaxArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class SoftmaxVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    A softmax-output-layer sibling of VectorizedMultiClassBackpropClassifierNetwork. Hidden layers stay plain ArrayLayer (sigmoid); only
    the output layer is a SoftmaxArrayLayer - the array-level analogue of
    SoftmaxMultiClassBackpropClassifierNetwork's own output_layer_cls-only override over
    BackpropNetworkBase.

    No hyperparameter and no extra constructor parameter, so nothing beyond this one
    class-attribute override is needed - __init__/randomized/save/load are all inherited
    unchanged from VectorizedMultiClassBackpropClassifierNetwork. classify_state/
    predict_probabilities are inherited unchanged too: unlike softmax's per-node counterpart
    (SoftmaxOutputLayer needs its own joint-normalization-aware predict_probabilities),
    SoftmaxArrayLayer.forward already does the joint normalization itself, so the network-level
    forward loop needs no special-casing at all.
    """

    output_layer_cls = SoftmaxArrayLayer
