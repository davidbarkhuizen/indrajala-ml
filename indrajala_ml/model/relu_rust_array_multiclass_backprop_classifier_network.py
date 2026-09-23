from __future__ import annotations

from indrajala_ml.model.relu_rust_array_layer import ReLURustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class ReLURustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    The Rust-matmul-backed counterpart to ReLUVectorizedMultiClassBackpropClassifierNetwork.
    Hidden layers are built from ReLURustArrayLayer; the output layer stays a plain RustArrayLayer (sigmoid) - the same
    hidden-layer-only split ReLUVectorizedMultiClassBackpropClassifierNetwork uses.

    No hyperparameter and no extra constructor parameter, so nothing beyond this one
    class-attribute override is needed - __init__/randomized/save/load are all inherited
    unchanged from RustArrayMultiClassBackpropClassifierNetwork.
    """

    hidden_layer_cls = ReLURustArrayLayer
