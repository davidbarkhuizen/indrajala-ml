from __future__ import annotations

from typing import Sequence

from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.state_layer import StateLayer


class BackpropLayer:
    """
    A layer of BackpropNodes, each fully connected to the given input layer - modeled on
    AssociationLayer, but with an explicit forward() pass: node.value() is a pure cache read,
    so the layer-level forward() is what actually populates every node's cached activation.
    """

    # override point for a layer whose nodes need a different per-node class (e.g.
    # SoftmaxOutputLayer's SoftmaxOutputNode) - a plain class attribute, not a constructor
    # parameter, since every node in a layer is always the same class and this keeps every
    # existing caller (BackpropNetworkBase, tests) unchanged
    _node_cls: type[BackpropNode] = BackpropNode

    def __init__(self, size: int, input_layer: StateLayer | "BackpropLayer") -> None:

        assert size >= 1, f"a layer must have at least 1 node; got size={size}"

        self.size: int = size

        self.input_layer: StateLayer | "BackpropLayer" = input_layer

        self.nodes: Sequence[BackpropNode] = [self._node_cls(input_nodes=self.input_layer.nodes) for _ in range(size)]

    def forward(self) -> None:
        for node in self.nodes:
            node.forward()

    def set_training_mode(self, training: bool) -> None:
        # a no-op by default - every existing layer type is unaffected and needs no change at
        # all. Only a sibling whose forward pass genuinely differs between training and
        # inference (e.g. DropoutLayer) overrides this to propagate the flag to its own nodes.
        pass

    # these five methods are BackpropNetworkBase's own per-node loops, extracted here so a
    # sibling layer with a different notion of "one weight-owning unit" than "one node" (e.g. a
    # convolutional layer sharing one kernel across many spatial-position nodes) can override
    # just these, once per layer, instead of the network reaching into layer.nodes directly
    def apply_gradients(self, learning_rate: float) -> None:
        for node in self.nodes:
            node.apply_gradient(learning_rate)

    def accumulate_gradients(self) -> None:
        for node in self.nodes:
            node.accumulate_gradient()

    def apply_accumulated_gradients(self, learning_rate: float, batch_size: int) -> None:
        for node in self.nodes:
            node.apply_accumulated_gradient(learning_rate, batch_size)

    def snapshot_state(self) -> list[tuple[list[float], float]]:
        return [(list(node.input_node_weights), node.bias) for node in self.nodes]

    def restore_state(self, layer_snapshot: list[tuple[list[float], float]]) -> None:
        for node, (weights, bias) in zip(self.nodes, layer_snapshot):
            node.update_input_weights(weights)
            node.bias = bias
