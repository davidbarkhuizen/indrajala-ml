from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.specs.layer_specs import Dense


class BinaryCrossEntropyBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    BackpropClassifierNetwork with binary cross-entropy loss instead of quadratic (MSE): only
    the output layer's spec differs (CrossEntropyOutputLayer), since cross-entropy's delta is as
    per-node as quadratic loss's.

    Not a drop-in improvement at BackpropClassifierNetwork's tuned learning rates: at the demos'
    learning_rate=1.0 it reached 91.87% mean training accuracy against 97.80% (10 seeds, a fixed XOR
    scenario), because the larger, undamped gradient overshoots. learning_rate=0.1 brought it to
    97.60%. Tune its learning rate separately, typically much lower. The ensembles don't use it;
    the toy result isn't assumed to carry over to MNIST.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, loss="cross_entropy")
