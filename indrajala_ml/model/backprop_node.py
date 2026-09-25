from __future__ import annotations

import math
from typing import Sequence

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

        # accumulated by accumulate_gradient() over a mini-batch, consumed and reset by
        # apply_accumulated_gradient()
        self._weight_gradient_accum: list[float] = [0.0 for _ in self.input_nodes]
        self._bias_gradient_accum: float = 0.0

    @property
    def bias(self) -> float:
        return self._offset

    @bias.setter
    def bias(self, value: float) -> None:
        self._offset = value

    def forward(self) -> float:
        # the only place the activation is computed; value() reads the cache
        self._activation = sigmoid(self.z())
        return self._activation

    def value(self) -> float:
        return self._activation

    def compute_output_delta(self, reference_value: float) -> None:
        a = self.value()
        self.delta = (a - reference_value) * a * (1.0 - a)

    def compute_hidden_delta(self, next_layer_nodes: Sequence["BackpropNode"], own_index: int) -> None:
        # every node in next_layer_nodes has this node at own_index in its input_node_weights,
        # since a BackpropLayer builds every node from the same input_layer.nodes
        a = self.value()
        downstream = sum(node.delta * node.input_node_weights[own_index] for node in next_layer_nodes)
        self.delta = downstream * a * (1.0 - a)

    def accumulate_gradient(self) -> None:
        for i, node in enumerate(self.input_nodes):
            self._weight_gradient_accum[i] += self.delta * node.value()
        self._bias_gradient_accum += self.delta

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.update_input_weights(
            [
                weight - learning_rate * accum / batch_size
                for weight, accum in zip(self.input_node_weights, self._weight_gradient_accum)
            ]
        )
        self.bias = self.bias - learning_rate * self._bias_gradient_accum / batch_size
        self._reset_gradient_accum()

    def _reset_gradient_accum(self) -> None:
        self._weight_gradient_accum = [0.0 for _ in self.input_nodes]
        self._bias_gradient_accum = 0.0

    def apply_gradient(self, learning_rate: float) -> None:
        # accumulate + apply at batch_size=1, through the overridable pair, so a subclass
        # (MomentumBackpropNode, L2RegularizedBackpropNode) needn't override this
        self.accumulate_gradient()
        self.apply_accumulated_gradient(learning_rate, 1)
