from __future__ import annotations

from indrajala_ml.model.dropout_array_layer import DropoutArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class DropoutVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The dropout sibling of VectorizedMultiClassBackpropClassifierNetwork. Hidden layers are built
    from DropoutArrayLayer (with
    drop_probability bound via a closure); the output layer stays the inherited plain ArrayLayer
    (sigmoid) - the array-level analogue of DropoutBackpropClassifierNetwork's own
    hidden_layer_cls-only override, matching DropoutNode's hidden-layer-only convention.

    drop_probability is a required constructor parameter, no default - the same posture
    DropoutBackpropClassifierNetwork's per-node counterpart already takes.

    _set_training_mode overrides the base class's no-op hook to actually toggle every hidden
    layer's own set_training_mode - the one real behavioral fork in this sibling, not a swapped
    layer class. self.hidden_layers (self.layers[:-1] - self.layers[-1] is always the output
    layer, per ArrayNetworkBase.__init__'s own construction order) is kept as a real attribute,
    not just a local slice, since it's also part of this class's own public surface (tests
    inspect it directly to confirm training mode resets between calls). learn/learn_batch's
    train/eval bracketing (set_training_mode(True)/try/finally around the forward pass only, not
    the whole method) is inherited from ArrayNetworkBase unchanged.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int, drop_probability: float) -> None:
        self.drop_probability = drop_probability
        self.hidden_layer_cls = lambda size, input_size: DropoutArrayLayer(size, input_size, drop_probability)
        super().__init__(layer_sizes, dimension, class_count)
        self.hidden_layers = self.layers[:-1]

    def _set_training_mode(self, training: bool) -> None:
        for layer in self.hidden_layers:
            layer.set_training_mode(training)

    def _extra_state(self) -> dict:
        return {"drop_probability": self.drop_probability}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {"drop_probability": state["drop_probability"]}
