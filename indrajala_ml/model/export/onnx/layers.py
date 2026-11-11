"""
Each layer kind's ONNX emitter (the export-and-hub workplan, D9, D12): the inference forward pass of
the layer array_layer_builder.py builds for a spec, over its input tensor, in the graph's dtype. A
walk over expand_specs mirrors _build_layers: a residual block's fork keeps its input for its add.

Stage 3's kinds: dense (every activation and output), batch norm over a flat layer, the residual
fork and add, and dropout (the identity at inference, D12). Each kind a later stage adds raises
NotImplementedError naming that stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
    ExpandedSpec,
    Fork,
    GatedDense,
    LayerNorm,
    Merge,
    Patches,
    Position,
    Split,
    StateInput,
    TokenMean,
)

# the kinds later stages export, by the stage that adds them
_LATER: dict[type[Any], int] = {
    ConvSpec: 4,
    PoolSpec: 4,
    Patches: 4,
    TokenMean: 4,
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


def batch_norm(g: Graph, spec: BatchNorm, x: str, names: Sequence[str], weights: Sequence[Any]) -> str:
    """
    A batch-norm layer over a flat layer at inference, its running averages':
    activate(gamma (x - mean) / sqrt(var + epsilon) + beta), as BatchNormArrayLayer._inference.
    """
    gamma, beta, mean, variance = (g.weight(name, array) for name, array in zip(names, weights, strict=True))
    deviation = g.op("Sqrt", [g.op("Add", [variance, g.constant(spec.epsilon)])])
    xhat = g.op("Div", [g.op("Sub", [x, mean]), deviation])
    return activate(g, spec.activation, g.op("Add", [g.op("Mul", [gamma, xhat]), beta]))


def flat_layers(
    g: Graph,
    specs: Sequence[ExpandedSpec],
    x: str,
    names: Sequence[Sequence[str]],
    weights: Sequence[Sequence[Any]],
) -> str:
    """The output of expanded specs over x, a flat (N, d) input: stage 3's layers."""
    forks: list[str] = []
    for spec, entry_names, entry in zip(specs, names, weights, strict=True):
        if isinstance(spec, Fork):
            forks.append(x)
        elif isinstance(spec, Add):
            x = g.op("Add", [x, forks.pop()])
        elif isinstance(spec, Dropout):
            continue
        elif isinstance(spec, Dense):
            x = dense(g, spec, x, entry_names, entry)
        elif isinstance(spec, BatchNorm):
            x = batch_norm(g, spec, x, entry_names, entry)
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
