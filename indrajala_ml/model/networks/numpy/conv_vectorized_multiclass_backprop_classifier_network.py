from __future__ import annotations

from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.networks.numpy.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.array_network_shapes import ArrayConvShape


class ConvVectorizedMultiClassBackpropClassifierNetwork(
    ArrayConvShape[FloatArray], VectorizedMultiClassBackpropClassifierNetwork
):
    """
    ArrayConvShape on the numpy backend: ConvArrayLayers and MaxPoolArrayLayers in front of
    ArrayLayers.
    """
