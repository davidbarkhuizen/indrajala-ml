from __future__ import annotations

from indrajala_ml.model.adam_array_layer import AdamArrayLayer
from indrajala_ml.model.adam_backprop_classifier_network import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class AdamVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The Adam-optimized sibling of VectorizedMultiClassBackpropClassifierNetwork. Both hidden
    layers and the output layer are built from AdamArrayLayer with beta1/beta2/epsilon already
    bound via a closure (hidden_layer_cls ==
    output_layer_cls, set as instance attributes before super().__init__() runs) - the same
    pattern momentum/L2's own array siblings use.

    beta1/beta2/epsilon default to Kingma & Ba's own published values, matching
    AdamBackpropClassifierNetwork's per-node counterpart.

    snapshot()/restore() intentionally cover only W/b, matching the base array-backed sibling's
    own contract unchanged (BackpropNetworkBase.snapshot/restore likewise only ever captured
    weights/bias, never a momentum/Adam node's own velocity/m/v/t), not a new gap this class
    introduces. A resumed-training scenario that needs m/v/t preserved across a snapshot/restore
    round trip would need its own extended envelope; no measurement in this codebase has used one
    so far.
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
            lambda size, input_size: AdamArrayLayer(size, input_size, beta1, beta2, epsilon)
        )
        super().__init__(layer_sizes, dimension, class_count)

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
    ) -> "AdamVectorizedMultiClassBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, class_count, beta1, beta2, epsilon)
        network.randomize()
        return network

    def _extra_state(self) -> dict:
        return {"beta1": self.beta1, "beta2": self.beta2, "epsilon": self.epsilon}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {"beta1": state["beta1"], "beta2": state["beta2"], "epsilon": state["epsilon"]}
