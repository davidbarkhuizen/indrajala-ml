from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.adam_layer import make_adam_layer_cls

# Kingma & Ba (2014)'s published defaults. Unlike momentum's coefficient these have defaults:
# in practice they are near-fixed constants, not a tuned knob.
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPSILON = 1e-8


class AdamBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    An Adam (Kingma & Ba, 2014) sibling of BackpropClassifierNetwork: a per-parameter adaptive
    learning rate from bias-corrected running estimates of each weight's gradient mean and variance
    (see make_adam_layer_cls), instead of momentum's single velocity term. Adam changes the weight
    update, which every trainable layer shares, so both hidden_layer_cls and output_layer_cls are
    Adam layers, set in __init__ before BackpropNetworkBase.__init__ runs.

    Single-output only, like the other weight-update siblings of the per-node family.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        layer_cls = make_adam_layer_cls(beta1, beta2, epsilon)
        self.hidden_layer_cls = layer_cls
        self.output_layer_cls = layer_cls
        super().__init__(layer_sizes, dimension, input_bounds)
