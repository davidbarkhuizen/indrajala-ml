from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.array_backend import RUST
from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.rust_array_layer import RustArrayLayer


class RustArrayNetworkBase(ArrayNetworkBase[pa.Array]):
    """
    ArrayNetworkBase with the Rust backend's array operations (array_backend.py) and RustArrayLayer
    as the default layer class.
    """

    hidden_layer_cls = RustArrayLayer
    output_layer_cls = RustArrayLayer

    backend = RUST
