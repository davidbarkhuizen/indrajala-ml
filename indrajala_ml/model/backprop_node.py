from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from indrajala_ml.model.base_node import AbstractNode, WeightedInputNode


def sigmoid(z: float) -> float:
    """
    The logistic sigmoid 1/(1+e^-z), returning 0.0 when math.exp(-z) overflows (z below about
    -709), which a drifting weighted sum can reach. Not the piecewise e^z/(1+e^z) form for z < 0:
    that avoids the overflow too, but rounds differently across the whole negative range, which
    would move every pinned training result. In the tail the direct formula's value is 0.0 in
    float64 anyway.
    """

    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0


class BackpropNode(WeightedInputNode):
    """
    A sigmoid neuron trained by gradient descent, unlike AssociationNode's hard step and
    minimum-disturbance rule. bias plays the role of AssociationNode's threshold, named for being an
    additive term rather than a cutoff.
    """

    # the WeightSet flags (layer_protocols.py)
    weights_decayed: ClassVar[bool] = True
    has_bias: ClassVar[bool] = True

    # AbstractNode's, per example
    example_fields: ClassVar[tuple[str, ...]] = ("_activation", "delta")

    def __init__(
        self,
        input_nodes: Sequence[AbstractNode],
        input_node_weights: Sequence[float] | None = None,
        bias: float = 0.0,
    ) -> None:
        super().__init__(input_nodes, input_node_weights, offset=bias)

        # set by forward(); no default, since value() before forward() is a caller bug
        self._activation: float

        # populated by compute_output_delta()/compute_hidden_delta() during the backward pass
        self.delta: float

        # accumulated by accumulate_gradient() over a mini-batch, consumed and reset by the
        # network's optimizer (python_optimizer.py)
        self.weight_gradient_accum: list[float] = [0.0 for _ in self.input_nodes]
        self.bias_gradient_accum: float = 0.0

    @property
    def bias(self) -> float:
        return self._offset

    @bias.setter
    def bias(self, value: float) -> None:
        self._offset = value

    # the WeightSet surface (layer_protocols.py), shared with ConvKernel
    @property
    def weights(self) -> Sequence[float]:
        return self.input_node_weights

    def set_weights(self, weights: list[float]) -> None:
        self.update_input_weights(weights)

    def forward(self) -> float:
        # the only place the activation is computed; value() reads the cache
        self._activation = sigmoid(self.z())
        return self._activation

    def value(self) -> float:
        return self._activation

    def compute_output_delta(self, reference_value: float) -> None:
        a = self.value()
        self.delta = (a - reference_value) * a * (1.0 - a)

    def compute_hidden_delta(self, downstream: float) -> None:
        # downstream: the next layer's downstream_sum at this node's index (a dense layer's
        # sum over its nodes of delta * the weight each applies to this node, NodeLayer.downstream_sum)
        a = self.value()
        self.delta = downstream * a * (1.0 - a)

    def accumulate_gradient(self) -> None:
        for i, node in enumerate(self.input_nodes):
            self.weight_gradient_accum[i] += self.delta * node.value()
        self.bias_gradient_accum += self.delta

    def reset_gradient_accum(self) -> None:
        self.weight_gradient_accum = [0.0 for _ in self.input_nodes]
        self.bias_gradient_accum = 0.0
