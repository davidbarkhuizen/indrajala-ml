from __future__ import annotations

from indrajala_ml.model.l2_array_layer import L2ArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class L2VectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The L2 (weight decay) regularized sibling of VectorizedMultiClassBackpropClassifierNetwork.

    l2_lambda is a required constructor parameter, no default - the same posture
    L2RegularizedBackpropClassifierNetwork's per-node counterpart already takes. Both hidden
    layers and the output layer are built from L2ArrayLayer with l2_lambda already bound via a
    closure (hidden_layer_cls/output_layer_cls set as instance attributes before
    super().__init__() runs), mirroring L2RegularizedBackpropClassifierNetwork's own
    hidden_layer_cls == output_layer_cls choice.

    snapshot()/restore() intentionally cover only W/b - L2 needs no extra per-parameter state to
    capture in the first place (see L2ArrayLayer's own docstring), so this is not a new gap the
    way it is for momentum/Adam's own array siblings.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, l2_lambda: float) -> None:
        self.l2_lambda = l2_lambda
        self.hidden_layer_cls = self.output_layer_cls = (
            lambda size, input_size: L2ArrayLayer(size, input_size, l2_lambda)
        )
        super().__init__(layer_sizes, dimension, class_count)

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
        l2_lambda: float,
    ) -> "L2VectorizedMultiClassBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, class_count, l2_lambda)
        network.randomize()
        return network

    def _extra_state(self) -> dict:
        return {"l2_lambda": self.l2_lambda}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {"l2_lambda": state["l2_lambda"]}
