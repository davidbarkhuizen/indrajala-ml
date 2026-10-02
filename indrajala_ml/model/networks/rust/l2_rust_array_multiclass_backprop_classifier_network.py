from __future__ import annotations

from indrajala_ml.model.networks.rust.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.update_rules import WeightDecay


class L2RustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    L2VectorizedMultiClassBackpropClassifierNetwork on the Rust backend.
    """

    hyperparameters = ("l2_lambda",)

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, l2_lambda: float) -> None:
        self.l2_lambda = l2_lambda
        super().__init__(layer_sizes, dimension, class_count)

    def _update_rule(self) -> WeightDecay:
        return WeightDecay(self.l2_lambda)
