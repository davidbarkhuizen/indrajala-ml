from __future__ import annotations

from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from indrajala_ml.model.specs.layer_specs import Dense


class ReLURustArrayBackpropClassifierNetwork(RustArrayBackpropClassifierNetwork):
    """
    ReLUArrayBackpropClassifierNetwork on the Rust backend: ReLURustArrayLayer hidden layers and
    a sigmoid RustArrayLayer output.
    """

    def _hidden_spec(self, size: int) -> Dense:
        return Dense(size, activation="relu")
