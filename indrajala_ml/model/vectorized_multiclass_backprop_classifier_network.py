from __future__ import annotations

from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.array_network_shapes import ArrayMultiClassShape
from indrajala_ml.model.numpy_array_network_base import NumpyArrayNetworkBase


class VectorizedMultiClassBackpropClassifierNetwork(ArrayMultiClassShape[FloatArray], NumpyArrayNetworkBase):
    """
    ArrayMultiClassShape on the numpy backend: MultiClassBackpropClassifierNetwork with one array
    operation per layer instead of one Python object per node. The base of every numpy multiclass
    sibling, and the reference the Rust networks are tested against; it is itself parity-checked
    against the pure-Python network.
    """
