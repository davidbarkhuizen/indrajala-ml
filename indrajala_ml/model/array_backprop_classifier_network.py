from __future__ import annotations

from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.array_network_shapes import ArraySingleOutputShape


class ArrayBackpropClassifierNetwork(ArraySingleOutputShape, ArrayNetworkBase):
    """
    ArraySingleOutputShape on the numpy backend: the sub-network of
    EnsembleArrayBackpropClassifierNetwork.

    randomize() is ArrayNetworkBase's fan-in-aware scheme (limit = 1/sqrt(fan_in)), not
    BackpropClassifierNetwork's per-dimension bounds-width scaling, which is tuned for 1-2D
    geometric problems; the ensemble trains on 784-pixel MNIST, where only fan-in-aware init has
    been measured to work.
    """
