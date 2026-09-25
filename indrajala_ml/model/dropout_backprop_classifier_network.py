from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.dropout_layer import make_dropout_layer_cls


class DropoutBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    A dropout (Srivastava et al., 2014) sibling of BackpropClassifierNetwork: each hidden node is
    zeroed with probability drop_probability on every training forward pass, and inactive at
    inference (see make_dropout_layer_cls). Only hidden_layer_cls is a dropout layer, set in
    __init__; a dropout node used as an output node raises. BackpropNetworkBase._set_training_mode
    switches the nodes' training flag for each training call.

    drop_probability is required, like momentum and l2_lambda (unlike Adam's defaults).
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
