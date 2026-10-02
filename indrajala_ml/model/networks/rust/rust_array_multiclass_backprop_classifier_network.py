from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.networks.rust.rust_array_network_base import RustArrayNetworkBase
from indrajala_ml.model.specs.array_network_shapes import ArrayMultiClassShape


class RustArrayMultiClassBackpropClassifierNetwork(ArrayMultiClassShape[pa.Array], RustArrayNetworkBase):
    """
    ArrayMultiClassShape on the Rust backend, the base of every Rust multiclass sibling. The Rust
    networks are the production backend; the numpy networks stay as the comparison point.
    """
