from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.networks.rust.rust_array_network_base import RustArrayNetworkBase
from indrajala_ml.model.specs.array_network_shapes import ArraySingleOutputShape


class RustArrayBackpropClassifierNetwork(ArraySingleOutputShape[pa.Array], RustArrayNetworkBase):
    """
    ArraySingleOutputShape on the Rust backend: the sub-network of
    EnsembleRustArrayBackpropClassifierNetwork.
    """
