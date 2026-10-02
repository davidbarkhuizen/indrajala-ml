from __future__ import annotations

from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.update_rules import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON, Adam


class AdamRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    AdamVectorizedMultiClassBackpropClassifierNetwork on the Rust backend. snapshot()/restore()
    cover only W/b, not the optimizer's m/v/t.
    """

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

    def _update_rule(self) -> Adam:
        return Adam(self.beta1, self.beta2, self.epsilon)
