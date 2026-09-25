from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode


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
    A one-node output layer of CrossEntropyOutputNode. Unlike SoftmaxOutputLayer it needs no
    forward() override: the node's own sigmoid is what this loss needs.
    """

    _node_cls = CrossEntropyOutputNode


class BinaryCrossEntropyBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    BackpropClassifierNetwork with binary cross-entropy loss instead of quadratic (MSE): only
    output_layer_cls differs, since cross-entropy's delta is as per-node as quadratic loss's.

    Not a drop-in improvement at BackpropClassifierNetwork's tuned learning rates: at the demos'
    learning_rate=1.0 it reached 91.87% mean training accuracy against 97.80% (10 seeds, a fixed XOR
    scenario), because the larger, undamped gradient overshoots. learning_rate=0.1 brought it to
    97.60%. Tune its learning rate separately, typically much lower. The ensembles don't use it;
    the toy result isn't assumed to carry over to MNIST.
    """

    output_layer_cls = CrossEntropyOutputLayer
