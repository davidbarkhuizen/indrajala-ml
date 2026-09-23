from __future__ import annotations

from typing import Sequence

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode


def relu_activation(z: float) -> float:
    """max(0, z) - shared verbatim with ConvUnit (conv_unit.py), the only other unit in this
    codebase using ReLU, so the formula has exactly one place to change."""
    return max(0.0, z)


def relu_hidden_delta(next_layer_nodes: Sequence["BackpropNode"], own_index: int, activation: float) -> float:
    """ReLU's derivative is 1 where z > 0 and 0 where z <= 0 (the same either-branch convention
    most practical implementations use; z == 0 exactly is measure-zero and doesn't matter in
    practice) - and since activation == max(0, z), activation > 0 exactly when z > 0, so the
    derivative can be read off the already-cached activation without storing or recomputing z
    separately. Shared verbatim with ConvUnit (conv_unit.py)."""
    downstream = sum(node.delta * node.input_node_weights[own_index] for node in next_layer_nodes)
    return downstream if activation > 0.0 else 0.0


class ReLUNode(BackpropNode):
    """
    A hidden-layer node using ReLU (max(0, z)) instead of BackpropNode's sigmoid - the modern
    default hidden-layer activation specifically because it doesn't saturate on its positive
    side at all, unlike sigmoid. Hidden-layer-only by convention, in this codebase and in the literature generally
    - an unbounded activation isn't suited to any of this codebase's output layers (plain
    sigmoid's (0,1) range, softmax's normalized probabilities, or cross-entropy's target-matching
    range), so this node is never used as an output_layer_cls's _node_cls.
    """

    def forward(self) -> float:
        self._activation = relu_activation(self.z())
        return self._activation

    def compute_output_delta(self, reference_value: float) -> None:
        raise NotImplementedError(
            "ReLUNode is a hidden-layer activation, not an output one - an unbounded activation "
            "isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_hidden_delta(self, next_layer_nodes: Sequence["BackpropNode"], own_index: int) -> None:
        self.delta = relu_hidden_delta(next_layer_nodes, own_index, self.value())


class ReLULayer(BackpropLayer):
    """
    A hidden layer of ReLUNodes - no forward() override needed (unlike SoftmaxOutputLayer's),
    since ReLU, like sigmoid, is a pure per-node computation with nothing to coordinate across
    siblings; only each node's own activation and delta formula differ from BackpropNode's.
    """

    _node_cls = ReLUNode
