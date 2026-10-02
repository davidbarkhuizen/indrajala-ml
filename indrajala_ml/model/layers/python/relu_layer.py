from __future__ import annotations

from indrajala_ml.model.layers.python.backprop_layer import BackpropLayer
from indrajala_ml.model.layers.python.backprop_node import BackpropNode


def relu_activation(z: float) -> float:
    """
    max(0, z), shared with ConvUnit.
    """
    return max(0.0, z)


def relu_delta(downstream: float, activation: float) -> float:
    """
    ReLU's derivative is 1 where z > 0 and 0 where z <= 0 (z == 0 exactly doesn't matter in
    practice). activation > 0 exactly when z > 0, so it reads the cached activation. Shared with
    ConvUnit, whose downstream sum comes from the next layer's downstream_sum().
    """
    return downstream if activation > 0.0 else 0.0


class ReLUNode(BackpropNode):
    """
    A hidden node with ReLU (max(0, z)) instead of sigmoid, which doesn't saturate on the positive
    side. Hidden only, as is usual: an unbounded activation fits none of the output layers (a
    sigmoid's (0, 1), softmax probabilities, cross-entropy targets).
    """

    def forward(self) -> float:
        self._activation = relu_activation(self.z())
        return self._activation

    def compute_output_delta(self, reference_value: float) -> None:
        raise NotImplementedError(
            "ReLUNode is a hidden-layer activation, not an output one - an unbounded activation "
            "isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_hidden_delta(self, downstream: float) -> None:
        self.delta = relu_delta(downstream, self.value())


class ReLULayer(BackpropLayer):
    """
    A hidden layer of ReLUNodes. ReLU is per node, so forward() isn't overridden (unlike
    SoftmaxOutputLayer).
    """

    _node_cls = ReLUNode
