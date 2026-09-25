from __future__ import annotations

from typing import Sequence

from indrajala_ml.model.base_node import AbstractNode
from indrajala_ml.model.conv_kernel import ConvKernel
from indrajala_ml.model.relu_layer import relu_activation, relu_delta


class ConvUnit(AbstractNode):
    """
    One output position of one conv channel. Not a BackpropNode, whose owned, rebound weight list
    doesn't suit a kernel many units share; an AbstractNode, so it can feed a dense BackpropLayer
    as any node does.

    forward()/compute_hidden_delta() use relu_layer's relu_activation/relu_delta, with weights from
    the shared ConvKernel. compute_hidden_delta takes the downstream sum from the next layer (see
    ConvLayer.compute_hidden_deltas), since a conv next layer has no node owning a weight at this
    unit's index.
    """

    def __init__(self, input_nodes: Sequence[AbstractNode], kernel: ConvKernel) -> None:

        assert len(input_nodes) == len(kernel.weights), (
            f"receptive field size ({len(input_nodes)}) must match the kernel's own weight "
            f"count ({len(kernel.weights)})"
        )

        self.input_nodes: Sequence[AbstractNode] = input_nodes
        self.kernel = kernel

        # set by forward()/compute_hidden_delta(); no default, as in BackpropNode
        self._activation: float
        self.delta: float

    def z(self) -> float:
        return (
            sum(node.value() * weight for node, weight in zip(self.input_nodes, self.kernel.weights))
            + self.kernel.bias
        )

    def forward(self) -> float:
        self._activation = relu_activation(self.z())
        return self._activation

    def value(self) -> float:
        return self._activation

    def compute_output_delta(self, reference_value: float) -> None:
        raise NotImplementedError(
            "ConvUnit is a hidden-layer unit, not an output one - see ReLUNode's identical "
            "guard (relu_layer.py), which this mirrors: an unbounded ReLU activation isn't "
            "suited to any of this codebase's output-layer contracts."
        )

    def compute_hidden_delta(self, downstream_sum: float) -> None:
        self.delta = relu_delta(downstream_sum, self.value())

    def accumulate_gradient(self) -> None:
        receptive_field_values = [node.value() for node in self.input_nodes]
        self.kernel.accumulate_gradient(self.delta, receptive_field_values)
