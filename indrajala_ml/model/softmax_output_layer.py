from __future__ import annotations

import math
from collections.abc import Sequence

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.state_layer import StateLayer


class SoftmaxOutputNode(BackpropNode):
    """
    An output node whose activation SoftmaxOutputLayer.forward() computes jointly with its layer and
    sets through activate(). forward() raises: a per-node sigmoid would be silently wrong.
    """

    def forward(self) -> float:
        raise NotImplementedError(
            "SoftmaxOutputNode.forward() must not be called directly - its activation is "
            "computed jointly across every node in its layer; see SoftmaxOutputLayer.forward()."
        )

    def activate(self, value: float) -> None:
        self._activation = value

    def compute_output_delta(self, reference_value: float) -> None:
        # softmax with cross-entropy: the softmax Jacobian cancels against the loss's
        # derivative, leaving no a*(1-a) factor
        self.delta = self.value() - reference_value


class SoftmaxOutputLayer(BackpropLayer):
    """
    An output layer with softmax activations, a_i = e^z_i / sum_j e^z_j over the layer, computed in
    forward(), since no node can compute its own.
    """

    _node_cls = SoftmaxOutputNode

    def __init__(self, size: int, input_layer: StateLayer | BackpropLayer) -> None:
        assert size >= 2, f"a softmax layer needs at least 2 nodes to normalize over; got size={size}"
        super().__init__(size, input_layer)

    def forward(self) -> None:
        nodes: Sequence[SoftmaxOutputNode] = self.nodes  # type: ignore[assignment]
        z_values = [node.z() for node in nodes]

        # subtract the max before exponentiating so no exponent overflows; the ratios are
        # unchanged, since e^(z-c) / sum(e^(z_j-c)) = e^z / sum(e^z_j)
        max_z = max(z_values)
        exp_values = [math.exp(z - max_z) for z in z_values]
        total = sum(exp_values)

        for node, exp_value in zip(nodes, exp_values):
            node.activate(exp_value / total)
