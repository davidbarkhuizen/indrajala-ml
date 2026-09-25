from __future__ import annotations

from indrajala_ml.model.l2_rust_array_layer import L2RustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class L2RustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    L2VectorizedMultiClassBackpropClassifierNetwork on the Rust backend: the same network, with
    L2RustArrayLayer in place of L2ArrayLayer (hidden and output layers, l2_lambda bound via a
    closure). l2_lambda is required, as there.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, l2_lambda: float) -> None:
        self.l2_lambda = l2_lambda
        self.hidden_layer_cls = self.output_layer_cls = (
            lambda size, input_size: L2RustArrayLayer(size, input_size, l2_lambda)
        )
        super().__init__(layer_sizes, dimension, class_count)

    def _extra_state(self) -> dict:
        return {"l2_lambda": self.l2_lambda}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {"l2_lambda": state["l2_lambda"]}
