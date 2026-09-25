from __future__ import annotations

from indrajala_ml.model.array_network_shapes import ArraySingleOutputShape
from indrajala_ml.model.rust_array_network_base import RustArrayNetworkBase


class RustArrayBackpropClassifierNetwork(ArraySingleOutputShape, RustArrayNetworkBase):
    """
    ArraySingleOutputShape on the Rust array core: ArrayBackpropClassifierNetwork with the Rust
    backend, hosting EnsembleRustArrayBackpropClassifierNetwork's sub-networks.
    CrossEntropyRustArrayBackpropClassifierNetwork subclasses this directly.
    """
