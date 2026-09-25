from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.momentum_layer import make_momentum_layer_cls


class MomentumBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    A momentum sibling of BackpropClassifierNetwork, adding the momentum term from Rumelhart,
    Hinton & Williams (1986)'s own generalized delta rule to every trainable layer's weight
    update (see make_momentum_layer_cls). Structurally this sets both hidden_layer_cls and
    output_layer_cls (unlike every other sibling class in this codebase, which needed only one)
    to the same momentum-configured layer class, as instance attributes set in __init__ before
    BackpropNetworkBase.__init__ runs - Python resolves self.hidden_layer_cls/self.output_layer_cls
    from the instance first, so this overrides the class-level default per instance without
    touching it for anyone else. Every other method (learn/_backward/randomize/snapshot/restore)
    is inherited unchanged.

    momentum is a required constructor parameter, not a keyword default, deliberately: unlike
    every other sibling class's fixed formula, there is no single coefficient this codebase's own
    measurements support recommending.
    Rumelhart et al.'s own cited value, momentum=0.9, robustly *hurt* relative to no momentum at
    all across a 10x learning-rate sweep on a fixed XOR scenario (best case: 92.50% vs the
    no-momentum baseline's 97.80%). A finer sweep of lower coefficients (0.3-0.7) at the original
    tuned learning rate, re-measured with enough seeds (15, not 5) to separate signal from noise,
    landed within 0.33 points of no momentum at all - a flat, statistically indistinguishable
    null, not a real effect. Committed anyway, as a genuine capability worth having on its own
    terms (e.g. for a future investigation under mini-batch gradients, where momentum's
    literature is more commonly validated - per-example online SGD's especially noisy individual
    gradients were the leading candidate explanation the finer sweep couldn't rule out) - not
    because any measurement here recommends turning it on.
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
