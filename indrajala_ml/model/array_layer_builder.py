"""
The numpy and Rust builder of layer specs (layer_specs.py): each spec to the backend's existing
layer class, in forward order, each layer's input shape the previous layer's output shape.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.array_protocols import ArrayNetworkLayer
from indrajala_ml.model.batch_norm_array_layer import BatchNormArrayLayer
from indrajala_ml.model.batch_norm_rust_array_layer import BatchNormRustArrayLayer
from indrajala_ml.model.conv_array_layer import ConvArrayLayer, LinearConvArrayLayer
from indrajala_ml.model.conv_front_end import ArrayFrontEndLayer
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer, LinearConvRustArrayLayer
from indrajala_ml.model.cross_entropy_array_layer import CrossEntropyArrayLayer
from indrajala_ml.model.cross_entropy_rust_array_layer import CrossEntropyRustArrayLayer
from indrajala_ml.model.dropout_array_layer import DropoutArrayLayer
from indrajala_ml.model.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.layer_specs import BatchNorm, Dense, InputShape, LayerSpec, validate_layer_specs
from indrajala_ml.model.linear_array_layer import LinearArrayLayer
from indrajala_ml.model.linear_rust_array_layer import LinearRustArrayLayer
from indrajala_ml.model.max_pool_array_layer import MaxPoolArrayLayer
from indrajala_ml.model.max_pool_rust_array_layer import MaxPoolRustArrayLayer
from indrajala_ml.model.relu_array_layer import ReLUArrayLayer
from indrajala_ml.model.relu_rust_array_layer import ReLURustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.softmax_array_layer import SoftmaxArrayLayer
from indrajala_ml.model.softmax_rust_array_layer import SoftmaxRustArrayLayer

LayerClass = Callable[..., ArrayNetworkLayer[Any]]
FrontEndLayerClass = Callable[..., ArrayFrontEndLayer[Any]]


@dataclass(frozen=True)
class ArrayLayerClasses:
    """One backend's layer class for each kind of layer a spec describes."""

    sigmoid: LayerClass
    relu: LayerClass
    dropout: LayerClass  # sigmoid, with dropout
    softmax: LayerClass  # output, cross-entropy
    cross_entropy: LayerClass  # sigmoid output, cross-entropy
    conv: FrontEndLayerClass
    pool: FrontEndLayerClass
    # batch norm's pair (the batch-norm workplan): a bias-free linear layer, then the norm layer
    linear: LayerClass
    batch_norm: LayerClass
    linear_conv: FrontEndLayerClass


LAYER_CLASSES = {
    "numpy": ArrayLayerClasses(
        sigmoid=ArrayLayer,
        relu=ReLUArrayLayer,
        dropout=DropoutArrayLayer,
        softmax=SoftmaxArrayLayer,
        cross_entropy=CrossEntropyArrayLayer,
        conv=ConvArrayLayer,
        pool=MaxPoolArrayLayer,
        linear=LinearArrayLayer,
        batch_norm=BatchNormArrayLayer,
        linear_conv=LinearConvArrayLayer,
    ),
    "rust": ArrayLayerClasses(
        sigmoid=RustArrayLayer,
        relu=ReLURustArrayLayer,
        dropout=DropoutRustArrayLayer,
        softmax=SoftmaxRustArrayLayer,
        cross_entropy=CrossEntropyRustArrayLayer,
        conv=ConvRustArrayLayer,
        pool=MaxPoolRustArrayLayer,
        linear=LinearRustArrayLayer,
        batch_norm=BatchNormRustArrayLayer,
        linear_conv=LinearConvRustArrayLayer,
    ),
}


def _dense_layer(classes: ArrayLayerClasses, spec: Dense, input_size: int) -> ArrayNetworkLayer[Any]:
    if spec.activation == "linear":
        return classes.linear(spec.size, input_size)
    if spec.dropout is not None:
        return classes.dropout(spec.size, input_size, spec.dropout)
    if spec.activation == "relu":
        return classes.relu(spec.size, input_size)
    if spec.activation == "softmax":
        return classes.softmax(spec.size, input_size)
    if spec.loss == "cross_entropy":
        return classes.cross_entropy(spec.size, input_size)
    return classes.sigmoid(spec.size, input_size)


def build_array_layers(
    specs: Sequence[LayerSpec], input_shape: InputShape, backend_name: str
) -> list[ArrayNetworkLayer[Any]]:
    """specs, validated (validate_layer_specs), as backend_name's layers over input_shape."""
    validate_layer_specs(specs)
    classes = LAYER_CLASSES[backend_name]

    layers: list[ArrayNetworkLayer[Any]] = []
    shape: InputShape = input_shape
    for spec in specs:
        if isinstance(spec, Dense):
            layers.append(_dense_layer(classes, spec, math.prod(shape)))
            shape = (spec.size,)
            continue
        if isinstance(spec, BatchNorm):
            # after a linear layer (validate_layer_specs), normalizing its features, or a conv
            # layer's channels over every position; the shape stays its linear layer's
            positions = shape[0] * shape[1] if len(shape) == 3 else 1
            layers.append(
                classes.batch_norm(math.prod(shape), spec.activation, spec.epsilon, spec.running_rate, positions)
            )
            continue

        assert len(shape) == 3, f"a conv or pool layer needs a (height, width, channels) input; got {shape}"
        height, width, channels = shape
        if isinstance(spec, ConvSpec):
            conv = classes.conv if spec.activation == "relu" else classes.linear_conv
            layer = conv(height, width, channels, spec.kernel_size, spec.channel_count, spec.stride)
        else:
            layer = classes.pool(height, width, channels, spec.pool_size, spec.stride)
        layers.append(layer)
        shape = (layer.out_height, layer.out_width, layer.channel_count)
    return layers
