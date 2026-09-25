from __future__ import annotations

from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.array_network_shapes import ArraySingleOutputShape
from indrajala_ml.model.numpy_array_network_base import NumpyArrayNetworkBase


class ArrayBackpropClassifierNetwork(ArraySingleOutputShape[FloatArray], NumpyArrayNetworkBase):
    """
    ArraySingleOutputShape on the numpy backend: the sub-network of
    EnsembleArrayBackpropClassifierNetwork.

    randomize() is ArrayNetworkBase's fan-in-aware scheme (limit = 1/sqrt(fan_in)), not
    BackpropClassifierNetwork's per-dimension bounds-width scaling, which is tuned for 1-2D
    geometric problems; the ensemble trains on 784-pixel MNIST, where only fan-in-aware init has
    been measured to work.
    """
