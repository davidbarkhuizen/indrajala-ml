from __future__ import annotations

from typing import ClassVar

from indrajala_ml.model.networks.python.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.specs.layer_specs import Dense


class DropoutBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    A dropout (Srivastava et al., 2014) sibling of BackpropClassifierNetwork: each hidden node is
    zeroed with probability drop_probability on every training forward pass, and inactive at
    inference (see make_dropout_layer_cls). Only the hidden layers drop out (_hidden_spec); a
    dropout node used as an output node raises. BackpropNetworkBase._set_training_mode
    switches the nodes' training flag for each training call.

    drop_probability is required, like momentum and l2_lambda (unlike Adam's defaults).
    """

    hyperparameters: ClassVar[tuple[str, ...]] = ("drop_probability",)

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        drop_probability: float,
    ) -> None:
        self.drop_probability = drop_probability
        super().__init__(layer_sizes, dimension, input_bounds)

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, dropout=self.drop_probability)
