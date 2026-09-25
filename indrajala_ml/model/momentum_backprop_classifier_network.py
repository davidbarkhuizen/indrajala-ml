from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.momentum_layer import make_momentum_layer_cls


class MomentumBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    A momentum sibling of BackpropClassifierNetwork: the momentum term of Rumelhart, Hinton &
    Williams (1986)'s generalized delta rule in every trainable layer's update (see
    make_momentum_layer_cls). Both hidden_layer_cls and output_layer_cls are momentum layers, set as
    instance attributes in __init__ before BackpropNetworkBase.__init__ runs.

    momentum is required: no measurement here supports a default. Rumelhart et al.'s 0.9 hurt
    across a 10x learning-rate sweep on a fixed XOR scenario (best 92.50% against 97.80% without
    momentum). 0.3-0.7 at the tuned learning rate, over 15 seeds, landed within 0.33 points of no
    momentum: a null. Per-example SGD's noisy gradients are the leading untested explanation;
    momentum is more often validated with mini-batches.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        momentum: float,
    ) -> None:
        layer_cls = make_momentum_layer_cls(momentum)
        self.hidden_layer_cls = layer_cls
        self.output_layer_cls = layer_cls
        super().__init__(layer_sizes, dimension, input_bounds)
