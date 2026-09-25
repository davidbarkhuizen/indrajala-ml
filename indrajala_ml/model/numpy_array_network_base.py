from __future__ import annotations

from indrajala_ml.model.array_backend import NUMPY
from indrajala_ml.model.array_layer import ArrayLayer, FloatArray
from indrajala_ml.model.array_network_base import ArrayNetworkBase


class NumpyArrayNetworkBase(ArrayNetworkBase[FloatArray]):
    """
    ArrayNetworkBase with the numpy backend's array operations (array_backend.py) and ArrayLayer
    as the default layer class: the base of every numpy network, as RustArrayNetworkBase is of
    the Rust ones.
    """

    hidden_layer_cls = ArrayLayer
    output_layer_cls = ArrayLayer

    backend = NUMPY
