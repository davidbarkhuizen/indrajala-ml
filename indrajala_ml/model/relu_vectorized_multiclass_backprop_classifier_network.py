from __future__ import annotations

from indrajala_ml.model.relu_array_layer import ReLUArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class ReLUVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    A ReLU-hidden-layer sibling of VectorizedMultiClassBackpropClassifierNetwork. Hidden layers are built from ReLUArrayLayer; the
    output layer stays a plain ArrayLayer (sigmoid) - the array-level analogue of
    ReLUBackpropClassifierNetwork's own hidden_layer_cls-only override over BackpropNetworkBase,
    matching ReLUNode's hidden-layer-only convention.

    No hyperparameter and no extra constructor parameter, so nothing beyond this one
    class-attribute override is needed - __init__/randomized/save/load are all inherited
    unchanged from VectorizedMultiClassBackpropClassifierNetwork.
    """

    hidden_layer_cls = ReLUArrayLayer
