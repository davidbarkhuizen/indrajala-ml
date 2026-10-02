from __future__ import annotations

from typing import ClassVar

from indrajala_ml.model.layer_specs import Dense
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork


class DropoutMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    The dropout sibling of MultiClassBackpropClassifierNetwork: dropout hidden layers at
    drop_probability (required), and a plain sigmoid output layer, as in
    DropoutBackpropClassifierNetwork. The pure-Python sibling of
    DropoutVectorizedMultiClassBackpropClassifierNetwork and
    DropoutRustArrayMultiClassBackpropClassifierNetwork.
    """

    hyperparameters: ClassVar[tuple[str, ...]] = ("drop_probability",)

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        drop_probability: float,
    ) -> None:
        self.drop_probability = drop_probability
        super().__init__(layer_sizes, dimension, input_bounds, class_count)

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, dropout=self.drop_probability)
