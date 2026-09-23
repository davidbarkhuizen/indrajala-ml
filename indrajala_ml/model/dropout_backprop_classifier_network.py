from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.dropout_layer import make_dropout_layer_cls


class DropoutBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    A dropout sibling of BackpropClassifierNetwork (Srivastava et al., 2014): each hidden node
    is independently zeroed with probability drop_probability on every training-time forward
    pass, inactive at inference (see make_dropout_layer_cls). Structurally this sets only
    hidden_layer_cls (not output_layer_cls, unlike momentum/L2/Adam) as an instance attribute in
    __init__, before BackpropNetworkBase.__init__ runs - dropout is a hidden-layer-only
    convention here, the same posture ReLUBackpropClassifierNetwork's own hidden_layer_cls-only
    shape takes (see make_dropout_node_cls's own compute_output_delta, which raises rather than
    silently allowing an output-layer use this class was never scoped to test). Every other
    method (learn/_backward/randomize/snapshot/restore) is inherited unchanged - none of them
    need to know a hidden layer's nodes have a training-mode flag, since
    BackpropNetworkBase._set_training_mode already handles that generically for any sibling that
    needs it, dropout included.

    drop_probability is a required constructor parameter, not a keyword default, deliberately:
    the same posture momentum/l2_lambda take, not Adam's beta1/beta2/epsilon.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        drop_probability: float,
    ) -> None:
        self.hidden_layer_cls = make_dropout_layer_cls(drop_probability)
        super().__init__(layer_sizes, dimension, input_bounds)

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        drop_probability: float,
    ) -> "DropoutBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, input_bounds, drop_probability)
        network.randomize()
        return network
