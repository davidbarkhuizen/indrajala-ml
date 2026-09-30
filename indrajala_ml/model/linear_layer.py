"""
The pure-Python linear layer before a batch-norm layer (the batch-norm workplan, D1 and D2), the
counterpart of linear_array_layer.py: each node's z, the weighted sum of its inputs as BackpropNode
sums it, with no bias and no activation. The norm layer's mean subtraction cancels a bias, and its
beta takes the bias's role (Ioffe & Szegedy 2015, § 3.2). The norm layer carries the activation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from indrajala_ml.model.backprop_layer import NodeLayer
from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.fan_in_aware_init import fan_in_aware_weights
from indrajala_ml.model.layer_specs import refuse_single_example


class LinearNode(BackpropNode):
    """
    A BackpropNode with the identity activation and no bias: its bias stays 0.0, and neither the
    optimizer (has_bias) nor a snapshot steps or records it. Its delta is its downstream from the
    norm layer, since the identity's derivative is 1 (LinearLayer.compute_hidden_deltas).
    """

    has_bias: ClassVar[bool] = False

    def forward(self) -> float:
        # WeightedInputNode.z()'s sum, without the offset
        self._activation = sum(
            [node.value() * weight for node, weight in zip(self.input_nodes, self.input_node_weights)]
        )
        return self._activation

    def compute_output_delta(self, reference_value: float) -> None:
        raise NotImplementedError("a linear layer is hidden, before a batch-norm layer")

    def compute_hidden_delta(self, next_layer_nodes: Sequence[BackpropNode], own_index: int) -> None:
        refuse_single_example(self)

    def accumulate_gradient(self) -> None:
        # BackpropNode's, without the bias
        for i, node in enumerate(self.input_nodes):
            self.weight_gradient_accum[i] += self.delta * node.value()


class LinearLayer(NodeLayer):
    """
    A layer of LinearNodes, trained through the layer-major batch path only (layer_major.py): a
    network with batch norm has no single-example step (D4).
    """

    _node_cls = LinearNode

    def compute_hidden_deltas(self, next_layer: Any) -> None:
        # the norm layer's downstream_sum is its dl/dx, this node's delta
        for own_index, node in enumerate(self.nodes):
            node.delta = next_layer.downstream_sum(own_index)

    def randomize_fan_in_aware(self) -> None:
        # per node, weights only: no bias to draw
        fan_in = len(self.input_layer.nodes)
        for node in self.nodes:
            node.update_input_weights(fan_in_aware_weights(fan_in))

    def snapshot_state(self) -> list[tuple[list[float]]]:
        return [(list(node.input_node_weights),) for node in self.nodes]

    def restore_state(self, layer_snapshot: Sequence[Sequence[list[float]]]) -> None:
        for node, (weights,) in zip(self.nodes, layer_snapshot):
            node.update_input_weights(list(weights))
