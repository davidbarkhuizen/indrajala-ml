from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.array_backend import RUST
from indrajala_ml.model.array_network_base import ArrayNetworkBase


class RustArrayNetworkBase(ArrayNetworkBase[pa.Array]):
    """
    ArrayNetworkBase with the Rust backend's array operations (array_backend.py) and layer classes
    (array_layer_builder.py).
    """

    backend = RUST
