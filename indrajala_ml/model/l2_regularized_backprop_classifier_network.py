from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.l2_regularization_layer import make_l2_layer_cls


class L2RegularizedBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    An L2 (weight decay) regularized sibling of BackpropClassifierNetwork, adding
    l2_lambda*weight to every weight's gradient (see make_l2_layer_cls) - bias is never
    regularized. Structurally this sets both hidden_layer_cls and output_layer_cls (the same
    pattern MomentumBackpropClassifierNetwork uses, for the same reason: this modifies the
    weight-update rule itself, shared by every trainable layer, not just one) as instance
    attributes in __init__, before BackpropNetworkBase.__init__ runs. Every other method
    (learn/_backward/randomize/snapshot/restore) is inherited unchanged.

    l2_lambda is a required constructor parameter, not a keyword default, matching
    MomentumBackpropClassifierNetwork's own posture.

    Measured directly on a small, fixed, finite proxy dataset (not the toy XOR target used for the other sibling
    classes - L2's whole purpose is generalization, which needs a dataset a network can actually
    overfit to): too small a coefficient (0.0001, 0.001) makes both training and held-out
    accuracy slightly *worse* than no regularization; l2_lambda=0.01 closes the train/test gap
    entirely, but by training accuracy falling to meet test accuracy (93.75% either way), not by
    test accuracy improving; l2_lambda=0.1 and above collapse the network to a constant
    prediction (confirmed directly: hidden weights decay to near-zero, max magnitude ~0.003). No
    coefficient tested improved held-out accuracy above the unregularized baseline on this proxy
    - plausibly because this codebase's small networks don't overfit severely enough for a
    weight-magnitude penalty to have much room to help.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        l2_lambda: float,
    ) -> None:
        layer_cls = make_l2_layer_cls(l2_lambda)
        self.hidden_layer_cls = layer_cls
        self.output_layer_cls = layer_cls
        super().__init__(layer_sizes, dimension, input_bounds)
