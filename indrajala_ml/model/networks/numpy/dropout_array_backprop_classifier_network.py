from __future__ import annotations

from indrajala_ml.model.layers.numpy.dropout_array_layer import DropoutArrayLayer
from indrajala_ml.model.networks.numpy.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.specs.layer_specs import Dense


class DropoutArrayBackpropClassifierNetwork(ArrayBackpropClassifierNetwork):
    """
    The dropout sibling of ArrayBackpropClassifierNetwork: DropoutArrayLayer hidden layers at
    drop_probability (required), and a plain sigmoid output layer, as in
    DropoutBackpropClassifierNetwork and DropoutVectorizedMultiClassBackpropClassifierNetwork.

    Its hyperparameters are keyword-only, after input_bounds, which ArraySingleOutputShape accepts
    and ignores. ArrayNetworkBase._set_training_mode switches the hidden layers' dropout on for the
    forward pass of each learn call; self.hidden_layers is part of the public surface.
    """

    hyperparameters = ("drop_probability",)

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
        *,
        drop_probability: float,
    ) -> None:
        self.drop_probability = drop_probability
        super().__init__(layer_sizes, dimension, input_bounds)
        hidden = self.layers[:-1]
        self.hidden_layers = [layer for layer in hidden if isinstance(layer, DropoutArrayLayer)]
        assert len(self.hidden_layers) == len(hidden)  # _hidden_spec: every hidden layer drops out

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, dropout=self.drop_probability)
