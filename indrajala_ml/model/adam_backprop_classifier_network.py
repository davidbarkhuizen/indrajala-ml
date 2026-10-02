from __future__ import annotations

from typing import ClassVar

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.specs.update_rules import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON, Adam


class AdamBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    An Adam (Kingma & Ba, 2014) sibling of BackpropClassifierNetwork: its optimizer applies the
    Adam rule to every trainable layer, a per-parameter adaptive learning rate from bias-corrected
    running estimates of each weight's gradient mean and variance, instead of momentum's single
    velocity term.

    Single-output only, like the other weight-update siblings of the per-node family.
    """

    hyperparameters: ClassVar[tuple[str, ...]] = ("beta1", "beta2", "epsilon")

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super().__init__(layer_sizes, dimension, input_bounds)

    def _update_rule(self) -> Adam:
        return Adam(self.beta1, self.beta2, self.epsilon)
