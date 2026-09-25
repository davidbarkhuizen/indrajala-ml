from __future__ import annotations

from indrajala_ml.model.cross_entropy_rust_array_layer import CrossEntropyRustArrayLayer
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork


class CrossEntropyRustArrayBackpropClassifierNetwork(RustArrayBackpropClassifierNetwork):
    """
    CrossEntropyArrayBackpropClassifierNetwork on the Rust backend, with a
    CrossEntropyRustArrayLayer output.
    """

    output_layer_cls = CrossEntropyRustArrayLayer
