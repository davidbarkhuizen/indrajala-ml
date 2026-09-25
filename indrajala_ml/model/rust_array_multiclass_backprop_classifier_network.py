from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.array_network_shapes import ArrayMultiClassShape
from indrajala_ml.model.rust_array_network_base import RustArrayNetworkBase


class RustArrayMultiClassBackpropClassifierNetwork(ArrayMultiClassShape[pa.Array], RustArrayNetworkBase):
    """
    ArrayMultiClassShape on the Rust backend, the base of every Rust multiclass sibling. The Rust
    networks are the production backend; the numpy networks stay as the comparison point.
    """
