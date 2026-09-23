from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.adam_layer import make_adam_layer_cls

# Kingma & Ba (2014)'s own published defaults - see adam_layer.make_adam_node_cls's docstring
# for why these, unlike momentum's coefficient, are given safe defaults rather than required
# explicitly: they're close to fixed algorithmic constants in virtually all real-world Adam
# usage, not a knob this codebase's own measurements have an opinion on.
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPSILON = 1e-8


class AdamBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    An Adam (Kingma & Ba, 2014) sibling of BackpropClassifierNetwork: a per-parameter adaptive
    learning rate driven by bias-corrected running estimates of each weight's own gradient mean
    and variance (see make_adam_layer_cls), rather than momentum's single shared velocity term.
    Structurally this sets both hidden_layer_cls and output_layer_cls (the same pattern
    MomentumBackpropClassifierNetwork/L2RegularizedBackpropClassifierNetwork use, for the same
    reason: this modifies the weight-update rule itself, shared by every trainable layer) as
    instance attributes in __init__, before BackpropNetworkBase.__init__ runs. Every other method
    (learn/_backward/randomize/snapshot/restore) is inherited unchanged.

    Scoped to BackpropClassifierNetwork only (single-output, binary), matching every prior
    weight-update-rule sibling's own launch scope; beta1/beta2/epsilon get safe defaults here
    rather than momentum's required-argument posture.
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

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
    ) -> "AdamBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, input_bounds, beta1, beta2, epsilon)
        network.randomize()
        return network
