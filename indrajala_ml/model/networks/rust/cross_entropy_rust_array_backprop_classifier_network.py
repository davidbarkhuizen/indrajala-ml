from __future__ import annotations

from indrajala_ml.model.networks.rust.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from indrajala_ml.model.specs.layer_specs import Dense


class CrossEntropyRustArrayBackpropClassifierNetwork(RustArrayBackpropClassifierNetwork):
    """
    CrossEntropyArrayBackpropClassifierNetwork on the Rust backend, with a
    CrossEntropyRustArrayLayer output.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, loss="cross_entropy")
