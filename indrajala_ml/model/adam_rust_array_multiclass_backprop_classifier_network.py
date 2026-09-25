from __future__ import annotations

from indrajala_ml.model.adam_backprop_classifier_network import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON
from indrajala_ml.model.adam_rust_array_layer import AdamRustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class AdamRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    AdamVectorizedMultiClassBackpropClassifierNetwork on the Rust backend: the same network, with
    AdamRustArrayLayer in place of AdamArrayLayer (hidden and output layers).

    snapshot()/restore() cover only W/b, not Adam's m/v/t (see
    AdamVectorizedMultiClassBackpropClassifierNetwork's docstring for why).
    """

    hidden_layer_cls = output_layer_cls = AdamRustArrayLayer
    hyperparameters = ("beta1", "beta2", "epsilon")

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super().__init__(layer_sizes, dimension, class_count)
