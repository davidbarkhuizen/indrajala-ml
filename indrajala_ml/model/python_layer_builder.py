"""
The pure-Python builder of layer specs (layer_specs.py): each spec to its existing pure-Python layer
class, in forward order, each layer wired to the previous one's nodes, its input shape the previous
layer's output shape. The numpy and Rust counterpart is array_layer_builder.py; both accept exactly
the specs validate_layer_specs does, so a spec list builds alike in all three implementations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from indrajala_ml.model.attention_layer import AttentionLayer
from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.batch_norm_layer import BatchNormLayer
from indrajala_ml.model.conv_layer import ConvLayer, ConvSpec
from indrajala_ml.model.cross_entropy_output_layer import CrossEntropyOutputLayer
from indrajala_ml.model.dropout_layer import make_dropout_layer_cls
from indrajala_ml.model.layer_norm_layer import LayerNormLayer
from indrajala_ml.model.linear_conv_layer import LinearConvLayer
from indrajala_ml.model.linear_layer import LinearLayer
from indrajala_ml.model.max_pool_layer import MaxPoolLayer, PoolSpec
from indrajala_ml.model.protocols.layer_protocols import InputLayer, TrainableLayer
from indrajala_ml.model.relu_layer import ReLULayer
from indrajala_ml.model.residual_layer import AddLayer, AffineLayer, ForkLayer
from indrajala_ml.model.softmax_output_layer import SoftmaxOutputLayer
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
from indrajala_ml.model.token_layer import PatchesLayer, PositionLayer, TokenDenseLayer, TokenMeanLayer


def _dense_layer(spec: Dense, input_layer: InputLayer) -> TrainableLayer:
    if spec.bias:
        return AffineLayer(size=spec.size, input_layer=input_layer)
    if spec.activation == "linear":
        return LinearLayer(size=spec.size, input_layer=input_layer)
    if spec.dropout is not None:
        return make_dropout_layer_cls(spec.dropout)(size=spec.size, input_layer=input_layer)
    if spec.activation == "relu":
        return ReLULayer(size=spec.size, input_layer=input_layer)
    if spec.activation == "softmax":
        return SoftmaxOutputLayer(size=spec.size, input_layer=input_layer)
    if spec.loss == "cross_entropy":
        return CrossEntropyOutputLayer(size=spec.size, input_layer=input_layer)
    return BackpropLayer(size=spec.size, input_layer=input_layer)


def _token_layer(
    spec: Patches | Position | LayerNorm | Attention | TokenMean, shape: Shape, input_layer: InputLayer
) -> TrainableLayer:
    if isinstance(spec, Patches):
        return PatchesLayer(input_layer, *image_shape(shape), spec.patch_size)
    tokens, features = token_shape(shape)
    if isinstance(spec, LayerNorm):
        return LayerNormLayer(input_layer, tokens, features, spec.epsilon)
    if isinstance(spec, Position):
        return PositionLayer(input_layer, tokens, features)
    if isinstance(spec, Attention):
        return AttentionLayer(input_layer, tokens, features)
    return TokenMeanLayer(input_layer, tokens, features)


def build_python_layers(
    specs: Sequence[LayerSpec], input_shape: InputShape, input_layer: InputLayer
) -> list[TrainableLayer]:
    """
    specs, validated (validate_layer_specs), as pure-Python layers reading input_layer, whose
    nodes are input_shape's flat layout: one per expanded spec (expand_specs), each residual block's
    fork wired to its add and its body's first layer.
    """
    validate_layer_specs(specs)
    shapes = spec_shapes(specs, input_shape)
    assert math.prod(input_shape) == len(input_layer.nodes), (
        f"input_shape {input_shape} doesn't match the input layer's {len(input_layer.nodes)} nodes"
    )

    layers: list[TrainableLayer] = []
    previous = input_layer
    # each open block's fork, and the fork whose body's first layer comes next
    forks: list[ForkLayer] = []
    opened: ForkLayer | None = None
    for spec, shape in zip(expand_specs(specs), shapes, strict=True):
        if isinstance(spec, Fork):
            layer: TrainableLayer = ForkLayer(previous)
            opened = layer
            forks.append(layer)
        elif isinstance(spec, Add):
            fork = forks.pop()
            layer = fork.add = AddLayer(previous, fork)
        elif isinstance(spec, Dense) and len(shape.input_shape) == 2:
            # a token-wise dense layer (D4): ReLU, or linear with a bias
            activation = "relu" if spec.activation == "relu" else "linear"
            layer = TokenDenseLayer(previous, spec.size, shape.input_shape[0], activation)
        elif isinstance(spec, Dense):
            layer = _dense_layer(spec, previous)
        elif isinstance(spec, Patches | Position | LayerNorm | Attention | TokenMean):
            layer = _token_layer(spec, shape.input_shape, previous)
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
        if opened is not None and not isinstance(spec, Fork):
            opened.body_first = layer
            opened = None
        layers.append(layer)
        previous = layer
    return layers
