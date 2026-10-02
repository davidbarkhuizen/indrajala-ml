from __future__ import annotations

from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.layers.numpy.array_layer import FloatArray


class NumpyArrayNetworkBase(ArrayNetworkBase[FloatArray]):
    """
    ArrayNetworkBase with the numpy backend's array operations (array_backend.py) and layer
    classes (array_layer_builder.py): the base of every numpy network, as RustArrayNetworkBase is
    of the Rust ones.
    """

    backend = NUMPY
