from __future__ import annotations

from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import Momentum


class MomentumRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    MomentumVectorizedMultiClassBackpropClassifierNetwork on the Rust backend.
    """

    hyperparameters = ("momentum",)

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, momentum: float) -> None:
        self.momentum = momentum
        super().__init__(layer_sizes, dimension, class_count)

    def _update_rule(self) -> Momentum:
        return Momentum(self.momentum)
