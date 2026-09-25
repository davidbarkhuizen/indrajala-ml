from __future__ import annotations

from indrajala_ml.model.cross_entropy_rust_array_layer import CrossEntropyRustArrayLayer
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork


class CrossEntropyRustArrayBackpropClassifierNetwork(RustArrayBackpropClassifierNetwork):
    """
    CrossEntropyArrayBackpropClassifierNetwork on the Rust backend: the same network, with
    CrossEntropyRustArrayLayer in place of CrossEntropyArrayLayer.

    No hyperparameter and no extra constructor parameter, so nothing beyond this one
    class-attribute override is needed - __init__/randomized/save/load/restore (including its
    plain-list tolerance for ensemble_train.py's multiprocessing path) are all inherited
    unchanged from RustArrayBackpropClassifierNetwork.
    """

    output_layer_cls = CrossEntropyRustArrayLayer
