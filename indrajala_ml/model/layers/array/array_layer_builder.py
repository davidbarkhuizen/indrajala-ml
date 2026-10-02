"""
The numpy and Rust builder of layer specs (layer_specs.py): each spec to the backend's existing
layer class, in forward order, each layer's input shape the previous layer's output shape.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from indrajala_ml.model.layers.numpy.array_layer import ArrayLayer
from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.layers.numpy.batch_norm_array_layer import BatchNormArrayLayer
from indrajala_ml.model.layers.numpy.conv_array_layer import ConvArrayLayer, LinearConvArrayLayer
from indrajala_ml.model.layers.numpy.cross_entropy_array_layer import CrossEntropyArrayLayer
from indrajala_ml.model.layers.numpy.dropout_array_layer import DropoutArrayLayer
from indrajala_ml.model.layers.numpy.layer_norm_array_layer import LayerNormArrayLayer
from indrajala_ml.model.layers.numpy.linear_array_layer import LinearArrayLayer
from indrajala_ml.model.layers.numpy.max_pool_array_layer import MaxPoolArrayLayer
from indrajala_ml.model.layers.numpy.relu_array_layer import ReLUArrayLayer
from indrajala_ml.model.layers.numpy.residual_array_layer import AddArrayLayer, AffineArrayLayer, ForkArrayLayer
from indrajala_ml.model.layers.numpy.softmax_array_layer import SoftmaxArrayLayer
from indrajala_ml.model.layers.numpy.token_array_layer import (
    PatchesArrayLayer,
    PositionArrayLayer,
    TokenDenseArrayLayer,
    TokenMeanArrayLayer,
)
from indrajala_ml.model.layers.python.conv_front_end import ArrayFrontEndLayer
from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.layers.rust.affine_rust_array_layer import AffineRustArrayLayer
from indrajala_ml.model.layers.rust.attention_rust_array_layer import AttentionRustArrayLayer
from indrajala_ml.model.layers.rust.batch_norm_rust_array_layer import BatchNormRustArrayLayer
from indrajala_ml.model.layers.rust.conv_rust_array_layer import ConvRustArrayLayer, LinearConvRustArrayLayer
from indrajala_ml.model.layers.rust.cross_entropy_rust_array_layer import CrossEntropyRustArrayLayer
from indrajala_ml.model.layers.rust.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.layers.rust.layer_norm_rust_array_layer import LayerNormRustArrayLayer
from indrajala_ml.model.layers.rust.linear_rust_array_layer import LinearRustArrayLayer
from indrajala_ml.model.layers.rust.max_pool_rust_array_layer import MaxPoolRustArrayLayer
from indrajala_ml.model.layers.rust.relu_rust_array_layer import ReLURustArrayLayer
from indrajala_ml.model.layers.rust.residual_rust_array_layer import AddRustArrayLayer, ForkRustArrayLayer
from indrajala_ml.model.layers.rust.rust_array_layer import RustArrayLayer
from indrajala_ml.model.layers.rust.softmax_rust_array_layer import SoftmaxRustArrayLayer
from indrajala_ml.model.layers.rust.token_rust_array_layer import (
    PatchesRustArrayLayer,
    PositionRustArrayLayer,
    TokenDenseRustArrayLayer,
    TokenMeanRustArrayLayer,
)
from indrajala_ml.model.protocols.array_protocols import ArrayNetworkLayer
from indrajala_ml.model.specs.layer_specs import (
    Add,
    Attention,
    BatchNorm,
    Dense,
    Fork,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    TokenMean,
    expand_specs,
)
from indrajala_ml.model.specs.spec_shapes import InputShape, Shape, image_shape, spec_shapes, token_shape
from indrajala_ml.model.specs.spec_validation import validate_layer_specs

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
    # a residual block's (the residual-connections workplan): the affine layer that ends a body,
    # the fork (size) and the add (its fork)
    affine: LayerClass
    fork: LayerClass
    add: LayerClass


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
        affine=AffineArrayLayer,
        fork=ForkArrayLayer,
        add=AddArrayLayer,
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
        affine=AffineRustArrayLayer,
        fork=ForkRustArrayLayer,
        add=AddRustArrayLayer,
    ),
}


@dataclass(frozen=True)
class TokenLayerClasses:
    """
    One backend's layer class for each layer of a patch model's and layer norm's (the layer-norm
    and attention workplan): the token-wise dense layer (ReLU or affine), patches, position, layer
    norm (over tokens or a flat layer), attention and the token mean.
    """

    token_dense: LayerClass
    patches: LayerClass
    position: LayerClass
    layer_norm: LayerClass
    attention: LayerClass
    token_mean: LayerClass


TOKEN_LAYER_CLASSES = {
    "numpy": TokenLayerClasses(
        token_dense=TokenDenseArrayLayer,
        patches=PatchesArrayLayer,
        position=PositionArrayLayer,
        layer_norm=LayerNormArrayLayer,
        attention=AttentionArrayLayer,
        token_mean=TokenMeanArrayLayer,
    ),
    "rust": TokenLayerClasses(
        token_dense=TokenDenseRustArrayLayer,
        patches=PatchesRustArrayLayer,
        position=PositionRustArrayLayer,
        layer_norm=LayerNormRustArrayLayer,
        attention=AttentionRustArrayLayer,
        token_mean=TokenMeanRustArrayLayer,
    ),
}


def _token_layer(classes: TokenLayerClasses, spec: LayerSpec, shape: Shape) -> ArrayNetworkLayer[Any] | None:
    # spec's layer if it is one of a patch model's or a layer norm, else None
    if isinstance(spec, Dense) and len(shape) == 2:
        return classes.token_dense(spec.size, shape[1], shape[0], spec.activation)
    if isinstance(spec, Patches):
        return classes.patches(*image_shape(shape), spec.patch_size)
    if isinstance(spec, LayerNorm):
        return classes.layer_norm(*token_shape(shape), spec.epsilon)
    if isinstance(spec, Position | Attention | TokenMean):
        tokens, features = token_shape(shape)
        layer_class = {Position: classes.position, Attention: classes.attention, TokenMean: classes.token_mean}
        return layer_class[type(spec)](tokens, features)
    return None


def _dense_layer(classes: ArrayLayerClasses, spec: Dense, input_size: int) -> ArrayNetworkLayer[Any]:
    if spec.bias:
        return classes.affine(spec.size, input_size)
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
    """
    specs, validated (validate_layer_specs), as backend_name's layers over input_shape: one per
    expanded spec (expand_specs), each residual block's fork wired to its add and to its body's
    first layer.
    """
    validate_layer_specs(specs)
    shapes = spec_shapes(specs, input_shape)
    classes = LAYER_CLASSES[backend_name]
    token_classes = TOKEN_LAYER_CLASSES[backend_name]

    layers: list[ArrayNetworkLayer[Any]] = []
    # each open block's fork, and the fork whose body's first layer comes next
    forks: list[Any] = []
    opened: Any = None
    for spec, shape in zip(expand_specs(specs), shapes, strict=True):
        input_size = math.prod(shape.input_shape)
        if isinstance(spec, Fork):
            opened = classes.fork(input_size)
            forks.append(opened)
            layers.append(opened)
            continue
        if isinstance(spec, Add):
            fork = forks.pop()
            fork.add = classes.add(fork)
            layers.append(fork.add)
            continue
        token_layer = _token_layer(token_classes, spec, shape.input_shape)
        if token_layer is not None:
            layers.append(token_layer)
        elif isinstance(spec, Dense):
            layers.append(_dense_layer(classes, spec, input_size))
        elif isinstance(spec, BatchNorm):
            layers.append(
                classes.batch_norm(
                    input_size, spec.activation, spec.epsilon, spec.running_rate, shape.positions, spec.group_size
                )
            )
        else:
            height, width, channels = image_shape(shape.input_shape)
            if isinstance(spec, ConvSpec):
                conv = classes.conv if spec.activation == "relu" else classes.linear_conv
                layers.append(conv(height, width, channels, spec.kernel_size, spec.channel_count, spec.stride))
            else:
                assert isinstance(spec, PoolSpec)
                layers.append(classes.pool(height, width, channels, spec.pool_size, spec.stride))
        if opened is not None:
            opened.body_first = layers[-1]
            opened = None
    return layers
