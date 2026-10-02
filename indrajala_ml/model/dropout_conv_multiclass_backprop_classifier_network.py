from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.specs.layer_specs import Dense


class DropoutConvMultiClassBackpropClassifierNetwork(ConvMultiClassBackpropClassifierNetwork):
    """
    The dropout sibling of ConvMultiClassBackpropClassifierNetwork: dropout dense hidden layers at
    drop_probability (required), and a plain sigmoid output layer, as in
    DropoutMultiClassBackpropClassifierNetwork. The conv front end doesn't drop out: _hidden_spec
    builds the dense hidden layers only. The pure-Python sibling of
    DropoutConvVectorizedMultiClassBackpropClassifierNetwork and
    DropoutConvRustArrayMultiClassBackpropClassifierNetwork.
    """

    hyperparameters: ClassVar[tuple[str, ...]] = ("drop_probability",)

    def __init__(
        self,
        input_height: int,
        input_width: int,
        conv_specs: Sequence[ConvSpec | PoolSpec],
        dense_layer_sizes: list[int],
        class_count: int,
        drop_probability: float,
    ) -> None:
        self.drop_probability = drop_probability
        super().__init__(input_height, input_width, conv_specs, dense_layer_sizes, class_count)

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, dropout=self.drop_probability)
