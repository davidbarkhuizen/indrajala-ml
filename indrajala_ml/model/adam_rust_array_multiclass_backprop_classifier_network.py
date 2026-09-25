from __future__ import annotations

from indrajala_ml.model.adam_backprop_classifier_network import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON
from indrajala_ml.model.adam_rust_array_layer import AdamRustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)


class AdamRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    AdamVectorizedMultiClassBackpropClassifierNetwork on the Rust backend: the same network, with
    AdamRustArrayLayer in place of AdamArrayLayer (hidden and output layers, beta1/beta2/epsilon
    bound via a closure).

    snapshot()/restore() cover only W/b, not Adam's m/v/t (see
    AdamVectorizedMultiClassBackpropClassifierNetwork's docstring for why).
    """

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
        self.hidden_layer_cls = self.output_layer_cls = (
            lambda size, input_size: AdamRustArrayLayer(size, input_size, beta1, beta2, epsilon)
        )
        super().__init__(layer_sizes, dimension, class_count)

    def _extra_state(self) -> dict:
        return {"beta1": self.beta1, "beta2": self.beta2, "epsilon": self.epsilon}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {"beta1": state["beta1"], "beta2": state["beta2"], "epsilon": state["epsilon"]}
