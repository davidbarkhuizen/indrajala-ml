from __future__ import annotations

from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.array_network_shapes import ArrayMultiClassShape


class VectorizedMultiClassBackpropClassifierNetwork(ArrayMultiClassShape, ArrayNetworkBase):
    """
    ArrayMultiClassShape on the numpy backend: MultiClassBackpropClassifierNetwork with one array
    operation per layer instead of one Python object per node. The base of every numpy multiclass
    sibling, and the reference the Rust networks are tested against; it is itself parity-checked
    against the pure-Python network.
    """
