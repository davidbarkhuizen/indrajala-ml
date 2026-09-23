from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode


class CrossEntropyOutputNode(BackpropNode):
    """
    A single-node output that overrides only compute_output_delta - binary cross-entropy loss's
    delta simplifies to activation - target, with no extra sigmoid-derivative factor (unlike
    BackpropNode.compute_output_delta's a*(1-a) term), the same simplification
    SoftmaxOutputNode.compute_output_delta uses for the multi-class case. Unlike softmax, a
    single output node's activation needs nothing from any sibling - forward() is inherited
    completely unchanged from BackpropNode, computing the same sigmoid(z()) it always has.
    """

    def compute_output_delta(self, reference_value: float) -> None:
        self.delta = self.value() - reference_value


class CrossEntropyOutputLayer(BackpropLayer):
    """
    A one-node output layer whose node is a CrossEntropyOutputNode - no forward() override
    needed (unlike SoftmaxOutputLayer's), since a single node's own independent sigmoid is
    already exactly what this loss function needs; only the node's delta computation differs.
    """

    _node_cls = CrossEntropyOutputNode


class BinaryCrossEntropyBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    A binary cross-entropy sibling of BackpropClassifierNetwork, aligned with the canonical
    binary classification treatment rather than that class's quadratic (MSE) loss - see its own
    docstring. Structurally this is a single class-attribute override
    (output_layer_cls = CrossEntropyOutputLayer), the same pattern
    SoftmaxMultiClassBackpropClassifierNetwork uses for the multi-class case: every other method
    (learn/_backward/randomize/randomized/snapshot/restore) is inherited unchanged and stays
    correct, since cross-entropy's delta is exactly as per-node-independent as quadratic loss's.

    Unlike the multi-class case, this is NOT a drop-in improvement at BackpropClassifierNetwork's
    existing tuned learning rates - measured directly: at the demo-tuned
    learning_rate=1.0 this class trains to a meaningfully *lower* training accuracy than
    BackpropClassifierNetwork on the same target (91.87% vs 97.80% mean over 10 seeds on a
    fixed XOR scenario), because cross-entropy's larger, undamped gradient overshoots at that
    step size. A learning-rate sweep on the same scenario found learning_rate=0.1 brings this
    class's mean training accuracy back up to 97.60%, matching the sibling's tuned performance -
    so this class needs its own, typically substantially lower, learning_rate tuned per use,
    not whatever value already works for BackpropClassifierNetwork.

    Deliberately not a replacement for BackpropClassifierNetwork, and not (yet) used by
    EnsembleBackpropClassifierNetwork's sub-networks or any existing demo - see the same docs
    entry for why extending this to real-MNIST ensemble training is separate, dedicated future
    work, not assumed to transfer from the toy problem measured here.
    """

    output_layer_cls = CrossEntropyOutputLayer
