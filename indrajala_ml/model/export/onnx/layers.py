"""
Each layer kind's ONNX emitter (the export-and-hub workplan, D9, D12): the inference forward pass of
the layer array_layer_builder.py builds for a spec, over its input tensor, in the graph's dtype. A
walk over expand_specs mirrors _build_layers: a residual block's fork keeps its input for its add.

Stage 3's kinds: dense (every activation and output), batch norm over a flat layer, the residual
fork and add, and dropout (the identity at inference, D12). Stage 4's: conv (im2col, D10), max
pool, batch norm per channel, patches and the token mean, over NCHW images, our row's own
channel-major order. Each kind a later stage adds raises NotImplementedError naming that stage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from indrajala_ml.model.export.onnx.functions import activate
from indrajala_ml.model.export.onnx.graph import Graph
from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.specs.layer_specs import (
    Add,
    Attention,
    BatchNorm,
    Branch,
    CrossAttention,
    Dense,
    Dropout,
    Embedding,
    Fork,
    GatedDense,
    LayerNorm,
    LayerSpec,
    Merge,
    Patches,
    Position,
    Split,
    StateInput,
    TokenMean,
    expand_specs,
)
from indrajala_ml.model.specs.spec_shapes import InputShape, Shape, spec_shapes
from indrajala_ml.model.specs.window_geometry import output_size

# the kinds later stages export, by the stage that adds them
_LATER: dict[type[Any], int] = {
    Embedding: 5,
    Position: 5,
    LayerNorm: 5,
    Attention: 5,
    GatedDense: 5,
    StateInput: 6,
    Split: 6,
    Branch: 6,
    Merge: 6,
    CrossAttention: 7,
}


def dense(g: Graph, spec: Dense, x: str, names: Sequence[str], weights: Sequence[Any]) -> str:
    """
    A flat dense layer, (N, d) to (N, size): array_layer_builder._dense_layer's choice of layer, in
    its order. W x + b alone for an affine layer (a residual body's end) and a given-loss output;
    W x for a linear one; a sigmoid for a dropout layer (dropout is training's only, D12); else the
    spec's activation, a sigmoid output's for the squared and cross-entropy losses.
    """
    W = g.weight(names[0], weights[0])
    if len(weights) == 1:
        return g.op("Gemm", [x, W], transB=1)
    z = g.op("Gemm", [x, W, g.weight(names[1], weights[1])], transB=1)
    if spec.bias or (spec.output and spec.loss == "given"):
        return z
    if spec.dropout is not None:
        return activate(g, "sigmoid", z)
    return activate(g, spec.activation, z)


def batch_norm(g: Graph, spec: BatchNorm, x: Tensor, names: Sequence[str], weights: Sequence[Any]) -> str:
    """
    A batch-norm layer at inference, its running averages': activate(gamma (x - mean) /
    sqrt(var + epsilon) + beta), as BatchNormArrayLayer._inference; per feature over a flat layer,
    per channel over an image (its parameters as (C, 1, 1), broadcast over NCHW's positions).
    """
    per_channel = len(x.dims) == 3
    gamma, beta, mean, variance = (
        g.weight(name, np.asarray(array).reshape(-1, 1, 1) if per_channel else array)
        for name, array in zip(names, weights, strict=True)
    )
    deviation = g.op("Sqrt", [g.op("Add", [variance, g.constant(spec.epsilon)])])
    xhat = g.op("Div", [g.op("Sub", [x.name, mean]), deviation])
    return activate(g, spec.activation, g.op("Add", [g.op("Mul", [gamma, xhat]), beta]))


def onnx_dims(shape: Shape) -> list[int]:
    """
    A spec shape's ONNX dimensions after the batch's: (d,) flat; (h, w, c) an image as (c, h, w),
    NCHW, our row's own channel-major order; (tokens, features) as they are. Each is a reshape of
    our flat row, so moving between them is a Reshape.
    """
    if len(shape) == 3:
        height, width, channels = shape
        return [channels, height, width]
    return list(shape)


@dataclass(frozen=True)
class Tensor:
    """A tensor in the graph and its dimensions after the batch's."""

    name: str
    dims: tuple[int, ...]


def as_shape(g: Graph, x: Tensor, shape: Shape) -> Tensor:
    """x reshaped to shape's ONNX dimensions, if it isn't already in them."""
    dims = tuple(onnx_dims(shape))
    if x.dims == dims:
        return x
    assert math.prod(x.dims) == math.prod(dims), f"{x.dims} doesn't reshape to {dims}"
    return Tensor(g.op("Reshape", [x.name, g.int64([-1, *dims])]), dims)


def conv(g: Graph, spec: ConvSpec, x: Tensor, names: Sequence[str], weights: Sequence[Any]) -> Tensor:
    """
    A conv layer as im2col and a MatMul (D10: onnxruntime has no float64 Conv), as
    ConvGeometryArrayLayer: each kernel offset's strided Slice, stacked to (N, C, k*k, oh, ow), as
    (N, P, C*k*k) columns in W's order, times W^T, plus b, then ReLU for a ReLU conv.
    """
    channels, height, width = x.dims
    k, stride = spec.kernel_size, spec.stride
    out_height, out_width = output_size(height, k, stride), output_size(width, k, stride)
    slices = [
        g.op(
            "Unsqueeze",
            [
                g.op(
                    "Slice",
                    [
                        x.name,
                        g.int64([i, j], "starts"),
                        g.int64([i + stride * (out_height - 1) + 1, j + stride * (out_width - 1) + 1], "ends"),
                        g.int64([2, 3], "axes"),
                        g.int64([stride, stride], "steps"),
                    ],
                ),
                g.int64([2], "axes"),
            ],
        )
        for i in range(k)
        for j in range(k)
    ]
    windows = g.op("Concat", slices, axis=2) if len(slices) > 1 else slices[0]
    positions = out_height * out_width
    columns = g.op(
        "Transpose",
        [g.op("Reshape", [windows, g.int64([-1, channels * k * k, positions])])],
        perm=[0, 2, 1],
    )
    W = g.op("Transpose", [g.weight(names[0], weights[0])], perm=[1, 0])
    z = g.op("MatMul", [columns, W])
    if len(weights) > 1:
        z = g.op("Add", [z, g.weight(names[1], weights[1])])
    if spec.activation == "relu":
        z = g.op("Relu", [z])
    planes = g.op("Transpose", [z], perm=[0, 2, 1])
    dims = (spec.channel_count, out_height, out_width)
    return Tensor(g.op("Reshape", [planes, g.int64([-1, *dims])]), dims)


def max_pool(g: Graph, spec: PoolSpec, x: Tensor) -> Tensor:
    """A max-pool layer over NCHW, no padding: onnxruntime runs MaxPool in float64."""
    channels, height, width = x.dims
    size = spec.pool_size
    stride = size if spec.stride is None else spec.stride  # non-overlapping windows by default
    pooled = g.op("MaxPool", [x.name], kernel_shape=[size, size], strides=[stride, stride])
    dims = (channels, output_size(height, size, stride), output_size(width, size, stride))
    return Tensor(pooled, dims)


def patches(g: Graph, spec: Patches, x: Tensor) -> Tensor:
    """An image's patches as tokens, as PatchesArrayLayer: (N, C, R, p, Cl, p) to (N, R*Cl, C*p*p)."""
    channels, height, width = x.dims
    p = spec.patch_size
    rows, columns = height // p, width // p
    grid = g.op("Reshape", [x.name, g.int64([-1, channels, rows, p, columns, p])])
    tokens = g.op("Transpose", [grid], perm=[0, 2, 4, 1, 3, 5])
    dims = (rows * columns, channels * p * p)
    return Tensor(g.op("Reshape", [tokens, g.int64([-1, *dims])]), dims)


def token_mean(g: Graph, x: Tensor) -> Tensor:
    """The mean over the tokens, (N, T, F) to (N, F)."""
    return Tensor(g.op("ReduceMean", [x.name, g.int64([1], "axes")], keepdims=0), x.dims[1:])


def layers(
    g: Graph,
    specs: Sequence[LayerSpec],
    input_shape: InputShape,
    x: Tensor,
    names: Sequence[Sequence[str]],
    weights: Sequence[Sequence[Any]],
) -> Tensor:
    """
    The output of specs over x: each expanded spec's emitter over x reshaped to the spec's input
    shape (spec_shapes), a residual block's add over its fork's input in the add's shape.
    """
    forks: list[Tensor] = []
    expanded = expand_specs(specs)
    shapes = spec_shapes(specs, input_shape)
    for spec, shape, entry_names, entry in zip(expanded, shapes, names, weights, strict=True):
        # a dense layer reads an image as our flat row; every other kind its input shape
        flat = isinstance(spec, Dense) and len(shape.input_shape) != 2
        x = as_shape(g, x, (math.prod(shape.input_shape),) if flat else shape.input_shape)
        if isinstance(spec, Fork):
            forks.append(x)
        elif isinstance(spec, Add):
            x = Tensor(g.op("Add", [x.name, as_shape(g, forks.pop(), shape.input_shape).name]), x.dims)
        elif isinstance(spec, Dropout):
            continue
        elif isinstance(spec, Dense) and len(x.dims) == 1:
            x = Tensor(dense(g, spec, x.name, entry_names, entry), (spec.size,))
        elif isinstance(spec, BatchNorm):
            x = Tensor(batch_norm(g, spec, x, entry_names, entry), x.dims)
        elif isinstance(spec, ConvSpec):
            x = conv(g, spec, x, entry_names, entry)
        elif isinstance(spec, PoolSpec):
            x = max_pool(g, spec, x)
        elif isinstance(spec, Patches):
            x = patches(g, spec, x)
        elif isinstance(spec, TokenMean):
            x = token_mean(g, x)
        else:
            raise NotImplementedError(unsupported(spec))
    return x


def unsupported(spec: object) -> str:
    """Why spec's kind doesn't export yet: the workplan stage that adds it."""
    kind = type(spec)
    stage = _LATER.get(kind)
    kind_name = kind.__name__
    if stage is None:
        return f"no ONNX emitter for {kind_name}"
    return f"ONNX export of {kind_name} comes in the export-and-hub workplan's stage {stage}"
