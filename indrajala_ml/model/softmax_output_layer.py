from __future__ import annotations

import math
from typing import Sequence

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.state_layer import StateLayer


class SoftmaxOutputNode(BackpropNode):
    """
    An output node whose activation is fixed jointly with every sibling in its layer via
    softmax, not computed independently from its own z() alone the way every other BackpropNode
    is - see SoftmaxOutputLayer.forward(), which computes every node's z(), normalizes them
    together, and pushes each node's share in via activate(). forward() is therefore
    deliberately unusable on its own here (unlike every other BackpropNode): calling it would
    silently compute a per-node sigmoid instead of the jointly-normalized softmax value
    everything else on this node (snapshot/restore-compatible weights, apply_gradient) assumes -
    raising catches that mistake immediately instead of computing a wrong answer quietly.
    """

    def forward(self) -> float:
        raise NotImplementedError(
            "SoftmaxOutputNode.forward() must not be called directly - its activation is "
            "computed jointly across every node in its layer; see SoftmaxOutputLayer.forward()."
        )

    def activate(self, value: float) -> None:
        self._activation = value

    def compute_output_delta(self, reference_value: float) -> None:
        # softmax + cross-entropy loss's output delta simplifies to exactly this - no extra
        # sigmoid-derivative factor the way BackpropNode.compute_output_delta's a*(1-a) term
        # needs, since softmax's own Jacobian cancels against cross-entropy's derivative in the
        # standard derivation
        self.delta = self.value() - reference_value


class SoftmaxOutputLayer(BackpropLayer):
    """
    An output layer whose activations are computed jointly via softmax
    (a_i = e^z_i / sum_j e^z_j across every node in this layer), rather than each node's own
    independent sigmoid(z()) the way BackpropLayer.forward() computes it - the layer-level
    coordination softmax needs is exactly why this overrides forward() instead of leaving
    SoftmaxOutputNode's own per-node forward() to do the work (it can't - see its own docstring).
    """

    _node_cls = SoftmaxOutputNode

    def __init__(self, size: int, input_layer: StateLayer | BackpropLayer) -> None:
        assert size >= 2, f"a softmax layer needs at least 2 nodes to normalize over; got size={size}"
        super().__init__(size, input_layer)

    def forward(self) -> None:
        nodes: Sequence[SoftmaxOutputNode] = self.nodes  # type: ignore[assignment]
        z_values = [node.z() for node in nodes]

        # standard numerically-stable softmax: subtract the max before exponentiating, so the
        # largest exponent evaluated is e^0=1 rather than e^z_max - which could otherwise
        # overflow the same way backprop_node.sigmoid's unguarded exp(-z) could (see that
        # function's docstring) - mathematically identical to the textbook formula, since
        # subtracting a constant from every z before exponentiating and normalizing leaves the
        # final ratios unchanged (e^(z-c) / sum(e^(z_j-c)) = e^z / sum(e^z_j) for any c)
        max_z = max(z_values)
        exp_values = [math.exp(z - max_z) for z in z_values]
        total = sum(exp_values)

        for node, exp_value in zip(nodes, exp_values):
            node.activate(exp_value / total)
