from __future__ import annotations

from indrajala_ml.model.array_network_shapes import ArrayConvShape
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.max_pool_array_layer import MaxPoolArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class ConvVectorizedMultiClassBackpropClassifierNetwork(ArrayConvShape, VectorizedMultiClassBackpropClassifierNetwork):
    """
    The numpy sibling of ConvMultiClassBackpropClassifierNetwork: ArrayConvShape on the numpy
    backend, with ConvArrayLayers and MaxPoolArrayLayers in front of ArrayLayers.
    """

    conv_layer_cls = ConvArrayLayer
    pool_layer_cls = MaxPoolArrayLayer
