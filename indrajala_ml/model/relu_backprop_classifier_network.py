from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.relu_layer import ReLULayer


class ReLUBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    BackpropClassifierNetwork with ReLU hidden layers (hidden_layer_cls = ReLULayer). The output
    stays sigmoid with quadratic loss (see ReLUNode for why ReLU is hidden-only), and randomize() is
    unchanged, so a measurement against the sigmoid network tests ReLU alone.

    At the demos' learning_rate=1.0 it trained to 74.03% mean training accuracy on a fixed XOR
    scenario, against 97.80% (10 seeds). Not dead units (1 of 8): ReLU's undamped gradient needs a
    smaller learning rate than sigmoid's a(1-a)-damped one, as with binary cross-entropy. At
    learning_rate=0.1 it reached 99.27%, above sigmoid's tuned result. No demo uses it; tune its
    learning rate wherever it's used.
    """

    hidden_layer_cls = ReLULayer
