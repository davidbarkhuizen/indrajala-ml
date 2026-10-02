"""
A residual block's layers in pure Python (the residual-connections workplan, stage 3; README,
Residual connections), the counterparts of residual_array_layer.py: the block's ForkLayer, which
passes its input on, the AddLayer, which adds it to the body's output, and the AffineLayer that ends
a body. The builder (python_layer_builder.py) wires each fork to its add and its body's first
layer.

The fork's and add's nodes cache their value and delta, per example (example_fields), so the
layer-major batch path (layer_major.py) can keep them in lanes. An add reads its fork's values in the
forward pass and a fork its add's deltas in the backward pass: each declares that layer
(forward_reads, backward_reads), and the batch path loads its lane too.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.base_node import AbstractNode
from indrajala_ml.model.layer_protocols import InputLayer, TrainableLayer, WeightSet
from indrajala_ml.pcg64 import Pcg64Generator


class AffineNode(BackpropNode):
    """
    A BackpropNode with the identity activation, W x + b (the residual-connections workplan, D4):
    its delta is its downstream, since the identity's derivative is 1. Its weights, bias and
    gradients are BackpropNode's. Hidden only.
    """

    def forward(self) -> float:
        self._activation = self.z()
        return self._activation

    def compute_output_delta(self, reference_value: float) -> None:
        raise NotImplementedError("an affine layer is hidden, at the end of a residual block's body")

    def compute_hidden_delta(self, downstream: float) -> None:
        self.delta = downstream


class AffineLayer(BackpropLayer):
    """A residual block's body's last layer: a layer of AffineNodes, drawn and stepped as a dense layer."""

    _node_cls = AffineNode


class PassNode(AbstractNode):
    """A fork's or an add's node: no weights, a value and a delta per example."""

    example_fields: ClassVar[tuple[str, ...]] = ("_activation", "delta")

    def __init__(self) -> None:
        # set by forward() and the backward pass; no default, as in BackpropNode
        self._activation: float
        self.delta: float

    def value(self) -> float:
        return self._activation


class ForkNode(PassNode):
    def __init__(self, input_node: AbstractNode) -> None:
        super().__init__()
        self.input_node = input_node

    def forward(self) -> None:
        self._activation = self.input_node.value()


class AddNode(PassNode):
    def __init__(self, body_node: AbstractNode, fork_node: ForkNode) -> None:
        super().__init__()
        self.body_node = body_node
        self.fork_node = fork_node

    def forward(self) -> None:
        # y + x: the body's output plus the block's input
        self._activation = self.body_node.value() + self.fork_node.value()


class ParameterFreeLayer[NodeT: PassNode]:
    """What a fork and an add share: no weights, so nothing to draw, accumulate, step or save."""

    nodes: Sequence[NodeT]

    # the layers besides the previous one whose nodes forward() reads, and besides the next one
    # whose nodes compute_hidden_deltas reads: the layer-major batch path loads their lanes too
    def forward_reads(self) -> tuple[TrainableLayer, ...]:
        return ()

    def backward_reads(self) -> tuple[TrainableLayer, ...]:
        return ()

    def set_training_mode(self, training: bool) -> None:
        pass

    def downstream_sum(self, own_index: int) -> float:
        # the gradient into the node before: this node's delta, the identity's
        return self.nodes[own_index].delta

    def accumulate_gradients(self) -> None:
        pass

    def weight_sets(self) -> Sequence[WeightSet]:
        return []

    def randomize_fan_in_aware(self, rng: Pcg64Generator) -> None:
        # D7: nothing drawn, so adding a block never shifts a later layer's draws
        pass

    def snapshot_state(self) -> list[Any]:
        return []

    def restore_state(self, layer_snapshot: Any) -> None:
        pass


class ForkLayer(ParameterFreeLayer[ForkNode]):
    """
    A residual block's first layer: its nodes pass the input layer's values on, and its delta is
    the sum of the two paths' gradients, the body's (its first layer's downstream_sum) and the
    identity's (the add's delta).
    """

    # set by the builder: the block's add, and its body's first layer
    add: AddLayer
    body_first: TrainableLayer

    def __init__(self, input_layer: InputLayer) -> None:
        self.input_layer = input_layer
        self.nodes = [ForkNode(node) for node in input_layer.nodes]
        self.size = len(self.nodes)

    def backward_reads(self) -> tuple[TrainableLayer, ...]:
        return (self.add,)

    def forward(self) -> None:
        for node in self.nodes:
            node.forward()

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
        # next_layer is body_first
        for own_index, (node, add_node) in enumerate(zip(self.nodes, self.add.nodes)):
            node.delta = next_layer.downstream_sum(own_index) + add_node.delta


class AddLayer(ParameterFreeLayer[AddNode]):
    """
    A residual block's last layer: each node the body's output plus the block's input, which its
    fork passed on. Its delta is the next layer's downstream_sum, and its downstream_sum is its
    delta, which the body's affine layer and the fork both read.
    """

    def __init__(self, input_layer: InputLayer, fork: ForkLayer) -> None:
        assert len(input_layer.nodes) == fork.size, "a residual block's output size is its input size (D5)"
        self.input_layer = input_layer
        self.fork = fork
        self.nodes = [AddNode(body_node, fork_node) for body_node, fork_node in zip(input_layer.nodes, fork.nodes)]
        self.size = len(self.nodes)

    def forward_reads(self) -> tuple[TrainableLayer, ...]:
        return (self.fork,)

    def forward(self) -> None:
        for node in self.nodes:
            node.forward()

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
        for own_index, node in enumerate(self.nodes):
            node.delta = next_layer.downstream_sum(own_index)
