from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.array_network_shapes import ArraySingleOutputShape
from indrajala_ml.model.rust_array_network_base import RustArrayNetworkBase


class RustArrayBackpropClassifierNetwork(ArraySingleOutputShape[pa.Array], RustArrayNetworkBase):
    """
    ArraySingleOutputShape on the Rust backend: the sub-network of
    EnsembleRustArrayBackpropClassifierNetwork.
    """
