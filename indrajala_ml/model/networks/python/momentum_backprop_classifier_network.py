from __future__ import annotations

from typing import ClassVar

from indrajala_ml.model.networks.python.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.specs.update_rules import Momentum


class MomentumBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    A momentum sibling of BackpropClassifierNetwork: its optimizer applies the Momentum rule,
    Goyal et al. 2017's eq. (9), to every trainable layer. It replaces Rumelhart, Hinton &
    Williams (1986)'s generalized delta rule, Δw(n) = η·δ·a + α·Δw(n-1) (the paper's eq. (10)),
    which folds the rate into the velocity and so needs a correction when the rate changes. At a
    constant rate the two are equivalent.

    momentum is required: no measurement here supports a default. Rumelhart et al.'s 0.9 hurt
    across a 10x learning-rate sweep on a fixed XOR scenario (best 92.50% against 97.80% without
    momentum). 0.3-0.7 at the tuned learning rate, over 15 seeds, landed within 0.33 points of no
    momentum: a null. Per-example SGD's noisy gradients are the leading untested explanation;
    momentum is more often validated with mini-batches.
    """

    hyperparameters: ClassVar[tuple[str, ...]] = ("momentum",)

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        momentum: float,
    ) -> None:
        self.momentum = momentum
        super().__init__(layer_sizes, dimension, input_bounds)

    def _update_rule(self) -> Momentum:
        return Momentum(self.momentum)
