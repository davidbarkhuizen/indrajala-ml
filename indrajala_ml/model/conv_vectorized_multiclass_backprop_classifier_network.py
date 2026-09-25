from __future__ import annotations

from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.array_network_shapes import ArrayConvShape
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.max_pool_array_layer import MaxPoolArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class ConvVectorizedMultiClassBackpropClassifierNetwork(
    ArrayConvShape[FloatArray], VectorizedMultiClassBackpropClassifierNetwork
):
    """
    ArrayConvShape on the numpy backend: ConvArrayLayers and MaxPoolArrayLayers in front of
    ArrayLayers.
    """

    conv_layer_cls = ConvArrayLayer
    pool_layer_cls = MaxPoolArrayLayer
