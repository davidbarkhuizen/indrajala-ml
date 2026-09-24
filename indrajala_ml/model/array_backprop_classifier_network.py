from __future__ import annotations

from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.array_network_shapes import ArraySingleOutputShape


class ArrayBackpropClassifierNetwork(ArraySingleOutputShape, ArrayNetworkBase):
    """
    The single-output numpy-array-backed sibling of FanInAwareBackpropClassifierNetwork:
    ArraySingleOutputShape on the numpy backend, hosting EnsembleArrayBackpropClassifierNetwork's
    sub-networks. CrossEntropyArrayBackpropClassifierNetwork subclasses this directly.

    randomize() (inherited from ArrayNetworkBase) implements only the fan-in-aware scheme (limit
    = 1/sqrt(fan_in)), skipping BackpropClassifierNetwork.randomize()'s own
    per-dimension-bounds-width scaling entirely: that scheme is "tuned for 1-2D geometric
    problems" (FanInAwareBackpropClassifierNetwork's own docstring), while this class exists for
    the ensemble's high-dimensional (784-pixel real-MNIST) use case, where fan-in-aware init is
    the only scheme ever measured to work.
    """
