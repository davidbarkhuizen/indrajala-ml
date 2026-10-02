from __future__ import annotations

from indrajala_ml.model.layers.python.backprop_layer import BackpropLayer
from indrajala_ml.model.layers.python.backprop_node import BackpropNode


class CrossEntropyOutputNode(BackpropNode):
    """
    An output node for binary cross-entropy: the delta is activation - target, without
    BackpropNode's a*(1-a) sigmoid-derivative factor (as SoftmaxOutputNode for the multiclass case).
    forward() is BackpropNode's sigmoid.
    """

    def compute_output_delta(self, reference_value: float) -> None:
        self.delta = self.value() - reference_value


class CrossEntropyOutputLayer(BackpropLayer):
    """
    An output layer of CrossEntropyOutputNode, one node per output (one for a single-output
    network). Unlike SoftmaxOutputLayer it needs no forward() override: the node's own sigmoid is
    what this loss needs.
    """

    _node_cls = CrossEntropyOutputNode
