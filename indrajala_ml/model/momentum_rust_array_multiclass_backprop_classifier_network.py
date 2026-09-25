from __future__ import annotations

from indrajala_ml.model.momentum_rust_array_layer import MomentumRustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class MomentumRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    MomentumVectorizedMultiClassBackpropClassifierNetwork on the Rust backend: the same network,
    with MomentumRustArrayLayer in place of MomentumArrayLayer (hidden and output layers, momentum
    bound via a closure). momentum is required, as there.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, momentum: float) -> None:
        self.momentum = momentum
        self.hidden_layer_cls = self.output_layer_cls = (
            lambda size, input_size: MomentumRustArrayLayer(size, input_size, momentum)
        )
        super().__init__(layer_sizes, dimension, class_count)

    def _extra_state(self) -> dict:
        return {"momentum": self.momentum}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {"momentum": state["momentum"]}
