from __future__ import annotations

from indrajala_ml.model.array_backend import RUST
from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.rust_array_layer import RustArrayLayer


class RustArrayNetworkBase(ArrayNetworkBase):
    """
    ArrayNetworkBase on the Rust array core: the same class, with the Rust backend's array
    operations (array_backend.py) and RustArrayLayer as the default layer class.
    """

    hidden_layer_cls: type = RustArrayLayer
    output_layer_cls: type = RustArrayLayer

    backend = RUST
