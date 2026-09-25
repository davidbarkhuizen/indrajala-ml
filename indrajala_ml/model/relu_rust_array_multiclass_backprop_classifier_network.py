from __future__ import annotations

from indrajala_ml.model.relu_rust_array_layer import ReLURustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class ReLURustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    ReLUVectorizedMultiClassBackpropClassifierNetwork on the Rust backend: ReLURustArrayLayer hidden
    layers and a sigmoid RustArrayLayer output.
    """

    hidden_layer_cls = ReLURustArrayLayer
