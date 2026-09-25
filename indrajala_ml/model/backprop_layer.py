from __future__ import annotations

from collections.abc import Sequence

from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.layer_protocols import InputLayer


class BackpropLayer:
    """
    A layer of BackpropNodes, each fully connected to the input layer. node.value() only reads a
    cached activation; forward() computes it.
    """

    # the node class, overridden by a sibling layer (e.g. SoftmaxOutputLayer)
    _node_cls: type[BackpropNode] = BackpropNode

    def __init__(self, size: int, input_layer: InputLayer) -> None:

        assert size >= 1, f"a layer must have at least 1 node; got size={size}"

        self.size: int = size

        self.input_layer: InputLayer = input_layer

        self.nodes: Sequence[BackpropNode] = [self._node_cls(input_nodes=self.input_layer.nodes) for _ in range(size)]

    def forward(self) -> None:
        for node in self.nodes:
            node.forward()

    def set_training_mode(self, training: bool) -> None:
        # a no-op except in a layer whose forward pass differs in training (DropoutLayer)
        pass

    def compute_hidden_deltas(self, next_layer: BackpropLayer) -> None:
        # a layer method so ConvLayer, whose units take a precomputed downstream sum, can
        # override it
        for own_index, node in enumerate(self.nodes):
            node.compute_hidden_delta(next_layer.nodes, own_index)

    def downstream_sum(self, own_index: int) -> float:
        # sum over this layer's nodes of delta * the weight each applies to the previous layer's
        # own_index-th node, in BackpropNode.compute_hidden_delta's order; ConvLayer overrides it
        # with its sparse, kernel-shared form
        return sum(node.delta * node.input_node_weights[own_index] for node in self.nodes)

    # the network calls these per layer, not per node, so a layer whose weights aren't one set
    # per node (ConvLayer's shared kernels) can override them
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
