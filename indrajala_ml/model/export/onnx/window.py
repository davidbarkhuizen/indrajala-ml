"""
The window-recurrent network's step (the export-and-hub workplan, stage 6, D10, D12): the
transducer's transition and output, one window a call, in float64 always (D10).

    inputs:  state (N, S), the snapped state before the window; window, its k token ids (N, k)
             int64, or an array window's observation, its input shape
    outputs: outputs, the window's (N, k, classes), or an array window's (N, classes);
             next_state (N, S), snapped; codes (N, factors) int64, each factor's code index (-1
             for a continuous factor); initial_state (1, S), the learned initial state snapped

The host loops it: state = initial_state for each row, then each window's next_state. Its parts
are RecurrenceArrayLayer's at inference, factor by factor: the attention pooling over the tokens
its factor reads (softmax of q . x_t, then the weighted sum), the GRU cell over [u, the other
factors' states] (cross) or u, then the nearest code by squared distance, the lowest index on a
tie (ArgMin's first), and that code's vector. A linear factor's codes are E = Q W, computed in the
graph from its W and Q as the network forms them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from indrajala_ml.model.export.onnx.graph import Graph
from indrajala_ml.model.export.onnx.layers import Tensor, onnx_dims, walk
from indrajala_ml.model.persistence.format3 import tensor_names
from indrajala_ml.model.specs.layer_specs import LayerSpec, StateInput, expand_specs, factor_reads
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.window_recurrence import WindowRecurrence, array_window, pooling_reads

WINDOW = "window"
STATE = "state"
NEXT_STATE = "next_state"
CODES = "codes"
INITIAL_STATE = "initial_state"
OUTPUTS = "outputs"


def pooling(g: Graph, tokens: str, q: str) -> str:
    """Attention pooling over (N, T, d): softmax over the tokens of q . x_t, then the tokens so weighted."""
    scores = g.op("MatMul", [tokens, q])  # (N, T)
    weights = g.op("Softmax", [scores], axis=-1)
    return g.op("Squeeze", [g.op("MatMul", [g.op("Unsqueeze", [weights, g.int64([1])]), tokens]), g.int64([1])])


def gru(g: Graph, u: str, c: str, W: str, U: str, b: str, size: int) -> str:
    """
    GRUArrayLayer: r = sigmoid(W_r u + U_r c + b_r), z likewise, h~ = tanh(W_h u + U_h (r * c) + b_h),
    h = z * c + (1 - z) * h~, the gates r, z, h in W's, U's and b's row blocks.
    """
    a = g.op("Add", [g.op("MatMul", [u, g.op("Transpose", [W], perm=[1, 0])]), b])
    U_rz = g.op("Slice", [U, g.int64([0]), g.int64([2 * size]), g.int64([0])])
    U_h = g.op("Slice", [U, g.int64([2 * size]), g.int64([3 * size]), g.int64([0])])
    a_rz = g.op("Slice", [a, g.int64([0]), g.int64([2 * size]), g.int64([1])])
    a_h = g.op("Slice", [a, g.int64([2 * size]), g.int64([3 * size]), g.int64([1])])
    rz = g.op("Add", [a_rz, g.op("MatMul", [c, g.op("Transpose", [U_rz], perm=[1, 0])])])
    r = g.op("Sigmoid", [g.op("Slice", [rz, g.int64([0]), g.int64([size]), g.int64([1])])])
    z = g.op("Sigmoid", [g.op("Slice", [rz, g.int64([size]), g.int64([2 * size]), g.int64([1])])])
    candidate = g.op(
        "Tanh", [g.op("Add", [a_h, g.op("MatMul", [g.op("Mul", [r, c]), g.op("Transpose", [U_h], perm=[1, 0])])])]
    )
    return g.op("Add", [g.op("Mul", [z, c]), g.op("Mul", [g.op("Sub", [g.constant(1.0), z]), candidate])])


def snap(g: Graph, h: str, E: str) -> tuple[str, str]:
    """h (N, s) to its nearest code's vector (N, s) and index (N,): squared distances, the lowest index on a tie."""
    differences = g.op("Sub", [g.op("Unsqueeze", [h, g.int64([1])]), g.op("Unsqueeze", [E, g.int64([0])])])
    distances = g.op("ReduceSum", [g.op("Mul", [differences, differences]), g.int64([2])], keepdims=0)
    index = g.op("ArgMin", [distances], axis=1, keepdims=0, select_last_index=0)
    return g.op("Gather", [E, index], axis=0), index


def _block(g: Graph, state: str, start: int, size: int) -> str:
    return g.op("Slice", [state, g.int64([start]), g.int64([start + size]), g.int64([1])])


def _others(g: Graph, state: str, start: int, size: int, total: int) -> str | None:
    # the other factors' states in factor order, c_(-g) (RecurrenceArrayLayer._others); None for one factor
    parts: list[str] = []
    if start > 0:
        parts.append(_block(g, state, 0, start))
    if start + size < total:
        parts.append(_block(g, state, start + size, total - start - size))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else g.op("Concat", parts, axis=1)


def _codes(g: Graph, names: Sequence[str], weights: Sequence[Any]) -> str:
    # a factor's code vectors: E, or a linear factor's Q W (LinearCodesArrayLayer._form)
    if len(weights) == 1:
        return g.weight(names[0], weights[0])
    W, Q = (g.weight(name, array) for name, array in zip(names, weights, strict=True))
    return g.op("MatMul", [Q, W])


def recurrence(
    g: Graph,
    recurrence_spec: WindowRecurrence,
    tokens: Sequence[Tensor],
    state: str,
    names: Sequence[Sequence[str]],
    weights: Sequence[Sequence[Any]],
) -> tuple[str, str]:
    """The next state (N, S) and each factor's code index (N, factors), -1 for a continuous one."""
    total = recurrence_spec.state_size
    entries = iter(zip(names, weights, strict=True))
    start = 0
    snapped: list[str] = []
    indices: list[str] = []
    for f, factor in enumerate(recurrence_spec.factors):
        size = factor.state_size
        (q_name,), (q,) = next(entries)
        cell_names, cell = next(entries)
        x = tokens[f]
        u = pooling(g, x.name, g.weight(q_name, q))
        own = _block(g, state, start, size)
        if recurrence_spec.cross:
            others = _others(g, state, start, size, total)
            if others is not None:
                u = g.op("Concat", [u, others], axis=1)
        W, U, b = (g.weight(name, array) for name, array in zip(cell_names, cell, strict=True))
        h = gru(g, u, own, W, U, b, size)
        if factor.codes is None:
            snapped.append(h)
            indices.append(g.op("Expand", [g.int64([-1], "continuous"), g.op("Shape", [state], start=0, end=1)]))
        else:
            code_names, code_weights = next(entries)
            vector, index = snap(g, h, _codes(g, code_names, code_weights))
            snapped.append(vector)
            indices.append(index)
        next(entries)  # the initial state, the initial_state output's
        start += size
    next_state = g.op("Concat", snapped, axis=1) if len(snapped) > 1 else snapped[0]
    codes = g.op("Concat", [g.op("Unsqueeze", [i, g.int64([1])]) for i in indices], axis=1)
    return next_state, codes


def initial_state(
    g: Graph, recurrence_spec: WindowRecurrence, names: Sequence[Sequence[str]], weights: Sequence[Sequence[Any]]
) -> str:
    """The learned initial state snapped (RecurrenceArrayLayer.first_state), (1, S)."""
    entries = list(zip(names, weights, strict=True))
    position = 0
    parts: list[str] = []
    for factor in recurrence_spec.factors:
        position += 2  # pooling, cell
        codes = None
        if factor.codes is not None:
            codes = entries[position]
            position += 1
        (c_name,), (c,) = entries[position]
        position += 1
        h = g.op("Unsqueeze", [g.weight(c_name, c), g.int64([0])])
        parts.append(h if codes is None else snap(g, h, _codes(g, *codes))[0])
    return parts[0] if len(parts) == 1 else g.op("Concat", parts, axis=1)


def step(
    g: Graph,
    specs: Sequence[LayerSpec],
    input_shape: InputShape,
    recurrence_spec: WindowRecurrence,
    weights: Sequence[Sequence[Any]],
) -> tuple[list[Any], list[Any]]:
    """The step's graph: its inputs' and outputs' value infos (the module's docstring)."""
    from indrajala_ml.model.export.onnx.dependencies import onnx

    int64 = onnx().TensorProto.INT64
    names = tensor_names(specs, recurrence_spec)
    expanded = expand_specs(specs)
    layer_count = len(expanded)
    tokens_window = not array_window(specs)
    dims = tuple(onnx_dims(input_shape))
    window = Tensor(WINDOW, dims)
    output, inputs = walk(g, specs, input_shape, window, names[:layer_count], weights[:layer_count], state=STATE)
    # the tokens each factor's pooling reads: its branch's end, or the output layer's or the first
    # StateInput's input (WindowRecurrentShape's _pooling_reads)
    ends = factor_reads(specs)
    if ends:
        reads = ends
    else:
        read = layer_count - 1
        if pooling_reads(specs, recurrence_spec) != len(specs) - 1:
            read = next(i for i, spec in enumerate(expanded) if isinstance(spec, StateInput))
        reads = [read] * len(recurrence_spec.factors)
    tokens = [_as_tokens(g, inputs[r]) for r in reads]
    next_state, codes = recurrence(g, recurrence_spec, tokens, STATE, names[layer_count:], weights[layer_count:])
    g.rename(output.name, OUTPUTS)
    g.rename(next_state, NEXT_STATE)
    g.rename(codes, CODES)
    g.rename(initial_state(g, recurrence_spec, names[layer_count:], weights[layer_count:]), INITIAL_STATE)
    size = recurrence_spec.state_size
    graph_inputs = [
        g.value_info(STATE, ["batch", size]),
        g.value_info(WINDOW, ["batch", *dims], int64 if tokens_window else None),
    ]
    graph_outputs = [
        g.value_info(OUTPUTS, ["batch", *output.dims]),
        g.value_info(NEXT_STATE, ["batch", size]),
        g.value_info(CODES, ["batch", len(recurrence_spec.factors)], int64),
        g.value_info(INITIAL_STATE, [1, size]),
    ]
    return graph_inputs, graph_outputs


def _as_tokens(g: Graph, x: Tensor) -> Tensor:
    # a pooling's (N, T, d): an array window's flat observation as one token
    if len(x.dims) == 2:
        return x
    dims = (1, int(np.prod(x.dims)))
    return Tensor(g.op("Reshape", [x.name, g.int64([-1, *dims])]), dims)
