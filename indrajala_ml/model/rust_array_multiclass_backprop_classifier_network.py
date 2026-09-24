from __future__ import annotations

from indrajala_ml.model.array_network_shapes import ArrayMultiClassShape
from indrajala_ml.model.rust_array_network_base import RustArrayNetworkBase


class RustArrayMultiClassBackpropClassifierNetwork(ArrayMultiClassShape, RustArrayNetworkBase):
    """
    ArrayMultiClassShape on the Rust array core: VectorizedMultiClassBackpropClassifierNetwork
    with `indrajala_math_rust.Array` via `RustArrayLayer` in place of numpy via `ArrayLayer`.
    Every Rust multiclass sibling subclasses this directly.

    This class is the intended production backend unconditionally - not contingent on beating
    VectorizedMultiClassBackpropClassifierNetwork's numpy benchmark, which stays on permanently
    as the comparison point, not a bar this class had to clear first.
    """
