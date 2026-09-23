from __future__ import annotations

from indrajala_ml.model.cross_entropy_rust_array_layer import CrossEntropyRustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class CrossEntropyRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    The Rust-matmul-backed counterpart to
    CrossEntropyVectorizedMultiClassBackpropClassifierNetwork. Hidden layers are built from plain
    RustArrayLayer (sigmoid); only the output layer is a CrossEntropyRustArrayLayer - the same
    split CrossEntropyVectorizedMultiClassBackpropClassifierNetwork uses. Unlike softmax's own
    output (jointly normalized), predict_probabilities does not sum to 1.0, and classify_state
    still picks the argmax (both inherited unchanged here).

    No hyperparameter and no extra constructor parameter, so nothing beyond this one
    class-attribute override is needed - __init__/randomized/save/load are all inherited
    unchanged from RustArrayMultiClassBackpropClassifierNetwork.
    """

    output_layer_cls = CrossEntropyRustArrayLayer
