"""
The shape walk over a list of layer specs (layer_specs.py): each spec's input and output shape over
the network's input shape, what every builder needs besides the choice of class.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from indrajala_ml.model.layers.python.conv_layer import ConvSpec
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
from indrajala_ml.model.specs.window_geometry import output_size, pool_stride

# a network's input: (dimension,) for a flat input, or (height, width, channels) for an image, whose
# flat layout is channel-major (conv_layer.py, conv_array_layer.py)
InputShape = tuple[int] | tuple[int, int, int]
ImageShape = tuple[int, int, int]
# a token sequence inside a network, (tokens, features), flat and token-major (D2): never an input
TokenShape = tuple[int, int]
Shape = InputShape | TokenShape


def image_shape(shape: Shape) -> ImageShape:
    """shape, which a conv or pool layer reads, as (height, width, channels)."""
    assert len(shape) == 3, f"a conv, pool or Patches layer needs a (height, width, channels) input; got {shape}"
    return shape


def token_shape(shape: Shape) -> TokenShape:
    """
    shape, which a layer norm or a patch model's token layer reads, as (tokens, features): a flat
    layer, or a conv front end's image, is one token.
    """
    return (shape[0], shape[1]) if len(shape) == 2 else (1, math.prod(shape))


@dataclass(frozen=True)
class SpecShape:
    """
    One spec's place in its list's shape walk (spec_shapes): the shape it reads, the shape it gives,
    and for a BatchNorm its positions, 1 after a dense layer or a conv layer's out_height *
    out_width (1 for every other spec).
    """

    input_shape: Shape
    output_shape: Shape
    positions: int = 1


def spec_shapes(specs: Sequence[LayerSpec | Fork | Add], input_shape: InputShape) -> list[SpecShape]:
    """
    Each of expand_specs(specs)' shapes over input_shape, in forward order, each spec's input shape
    the previous one's output shape: what every builder needs besides the choice of class. A
    BatchNorm keeps its linear layer's shape (validate_layer_specs), and a Fork and an Add their
    input's. A residual block's input is flat (D2) or tokens, and its Add's input is its Fork's
    (D5): both are checked here, where the shapes are known, as is a patch size that doesn't divide
    the image, and an Attention's heads that don't divide its token width when it has no key_size. A
    Dense over tokens acts on each (the layer-norm and attention workplan, D4), and a
    Position, a LayerNorm and an Attention keep their input's shape (a LayerNorm after a conv front
    end normalizes the flat image as one token). The specs' own arguments are checked by the layers
    built from them, not here.
    """
    shapes: list[SpecShape] = []
    shape: Shape = input_shape
    # each open block's input shape, its Fork's
    forks: list[Shape] = []
    for spec in expand_specs(specs):
        if isinstance(spec, Dense):
            shapes.append(SpecShape(shape, (shape[0], spec.size) if len(shape) == 2 else (spec.size,)))
        elif isinstance(spec, BatchNorm):
            positions = shape[0] * shape[1] if len(shape) == 3 else 1
            shapes.append(SpecShape(shape, shape, positions))
        elif isinstance(spec, Fork):
            assert len(shape) in (1, 2), (
                f"a residual block is dense or over tokens (D2): its input is flat or tokens; got {shape}"
            )
            forks.append(shape)
            shapes.append(SpecShape(shape, shape))
        elif isinstance(spec, Add):
            fork = forks.pop()
            assert shape == fork, (
                f"a residual block's output size is its input size (D5): its body ends in {shape}, not {fork}"
            )
            shapes.append(SpecShape(shape, shape))
        elif isinstance(spec, Patches):
            height, width, channels = image_shape(shape)
            size = spec.patch_size
            assert height % size == 0 and width % size == 0, (
                f"a patch size divides the image (the layer-norm and attention workplan, D3); got {spec!r} over {shape}"
            )
            shapes.append(SpecShape(shape, ((height // size) * (width // size), size * size * channels)))
        elif isinstance(spec, Attention):
            _, features = token_shape(shape)
            assert spec.key_size is not None or features % spec.heads == 0, (
                f"an Attention without a key_size splits the token width among its heads, so its heads divide it "
                f"(the multi-head attention workplan, D3); got {spec!r} over tokens of {features} features"
            )
            shapes.append(SpecShape(shape, shape))
        elif isinstance(spec, Position | LayerNorm):
            shapes.append(SpecShape(shape, shape))
        elif isinstance(spec, TokenMean):
            assert len(shape) == 2, f"a TokenMean reads tokens; got {shape}"
            shapes.append(SpecShape(shape, (shape[1],)))
        else:
            height, width, channels = image_shape(shape)
            if isinstance(spec, ConvSpec):
                window, stride, channels = spec.kernel_size, spec.stride, spec.channel_count
            else:
                window, stride = spec.pool_size, pool_stride(spec.pool_size, spec.stride)
            output = (output_size(height, window, stride), output_size(width, window, stride), channels)
            shapes.append(SpecShape(shape, output))
        shape = shapes[-1].output_shape
    return shapes
