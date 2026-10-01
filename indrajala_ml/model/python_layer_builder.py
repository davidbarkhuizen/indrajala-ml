"""
The pure-Python builder of layer specs (layer_specs.py): each spec to its existing pure-Python layer
class, in forward order, each layer wired to the previous one's nodes, its input shape the previous
layer's output shape. The numpy and Rust counterpart is array_layer_builder.py; both accept exactly
the specs validate_layer_specs does, so a spec list builds alike in all three implementations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.batch_norm_layer import BatchNormLayer
from indrajala_ml.model.conv_layer import ConvLayer, ConvSpec
from indrajala_ml.model.cross_entropy_output_layer import CrossEntropyOutputLayer
from indrajala_ml.model.dropout_layer import make_dropout_layer_cls
from indrajala_ml.model.layer_protocols import InputLayer, TrainableLayer
from indrajala_ml.model.layer_specs import (
    BatchNorm,
    Dense,
    InputShape,
    LayerSpec,
    image_shape,
    spec_shapes,
    validate_layer_specs,
)
from indrajala_ml.model.linear_conv_layer import LinearConvLayer
from indrajala_ml.model.linear_layer import LinearLayer
from indrajala_ml.model.max_pool_layer import MaxPoolLayer, PoolSpec
from indrajala_ml.model.relu_layer import ReLULayer
from indrajala_ml.model.softmax_output_layer import SoftmaxOutputLayer


def _dense_layer(spec: Dense, input_layer: InputLayer) -> TrainableLayer:
    if spec.dropout is not None:
        return make_dropout_layer_cls(spec.dropout)(size=spec.size, input_layer=input_layer)
    if spec.activation == "relu":
        return ReLULayer(size=spec.size, input_layer=input_layer)
    if spec.activation == "linear":
        return LinearLayer(size=spec.size, input_layer=input_layer)
    if spec.activation == "softmax":
        return SoftmaxOutputLayer(size=spec.size, input_layer=input_layer)
    if spec.loss == "cross_entropy":
        return CrossEntropyOutputLayer(size=spec.size, input_layer=input_layer)
    return BackpropLayer(size=spec.size, input_layer=input_layer)


def build_python_layers(
    specs: Sequence[LayerSpec], input_shape: InputShape, input_layer: InputLayer
) -> list[TrainableLayer]:
    """specs, validated (validate_layer_specs), as pure-Python layers reading input_layer, whose
    nodes are input_shape's flat layout."""
    validate_layer_specs(specs)
    assert math.prod(input_shape) == len(input_layer.nodes), (
        f"input_shape {input_shape} doesn't match the input layer's {len(input_layer.nodes)} nodes"
    )

    layers: list[TrainableLayer] = []
    previous = input_layer
    for spec, shape in zip(specs, spec_shapes(specs, input_shape), strict=True):
        if isinstance(spec, Dense):
            layer: TrainableLayer = _dense_layer(spec, previous)
        elif isinstance(spec, BatchNorm):
            layer = BatchNormLayer(
                previous, spec.activation, spec.epsilon, spec.running_rate, shape.positions, spec.group_size
            )
        else:
            height, width, channels = image_shape(shape.input_shape)
            if isinstance(spec, ConvSpec):
                conv_cls = LinearConvLayer if spec.activation == "linear" else ConvLayer
                layer = conv_cls(
                    input_layer=previous,
                    input_height=height,
                    input_width=width,
                    kernel_size=spec.kernel_size,
                    channel_count=spec.channel_count,
                    stride=spec.stride,
                    input_channels=channels,
                )
            else:
                assert isinstance(spec, PoolSpec)
                layer = MaxPoolLayer(
                    input_layer=previous,
                    input_height=height,
                    input_width=width,
                    input_channels=channels,
                    pool_size=spec.pool_size,
                    stride=spec.stride,
                )
        layers.append(layer)
        previous = layer
    return layers
