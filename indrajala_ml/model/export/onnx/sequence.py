"""
The sequence layers' ONNX emitters (the export-and-hub workplan, stage 5): embedding, position,
layer norm, token-wise dense layers and outputs, the gated FFN, and attention, over tokens
(N, T, F).

- **Segments come from the ids, in the graph,** as segments.segment_context derives them in the
  network: a token's segment is 1 + the separators before it (or, with separator_opens, the
  separators up to it, at least 1), a pad's 0; its position counts from its segment's first token.
  The graph's only input is the ids (D12 amended: no separate segments input).
- **Attention is decomposed** (D11 amended): Q K^T divided by sqrt(d_k), the disallowed scores -inf
  through Where, Softmax, then V, as Masking allows them: causal, by segment (another segment's or
  a pad's key masked, never the query's own column), grouped key/value heads expanded to the
  heads. Every rule is ours to check, in both evaluators and both dtypes; the fused Attention op is
  a speed-up to revisit once the decomposition is the reference.
- **The scale is a vector, not a scalar:** onnxruntime's MatMulScaleFusion would fold a scalar
  divisor into FusedMatMul's float32 alpha, 3e-8 of every score in a float64 graph.
- **Rotary is decomposed** (D10, D11): each head's halves rotated by rotary_table's cos and sin of
  positions 0 to T - 1, as the network does, constants of the graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from indrajala_ml.model.export.onnx.functions import activate
from indrajala_ml.model.export.onnx.graph import Graph
from indrajala_ml.model.specs.layer_specs import Attention, CrossAttention, Dense, Embedding, GatedDense, LayerNorm
from indrajala_ml.model.specs.rotary_table import rotary_table


@dataclass(frozen=True)
class Segments:
    """A batch's segments and positions in the graph, (N, T) int64 each, from its ids."""

    segments: str
    positions: str


def segments(g: Graph, ids: str, spec: Embedding, tokens: int) -> Segments:
    """ids' segments and positions (segments.segment_context's), (N, T) int64."""
    is_separator = g.op("Equal", [ids, g.int64(spec.separator, "separator")])
    is_pad = g.op("Equal", [ids, g.int64(spec.pad, "pad")])
    separators = g.op("Cast", [is_separator], to=_INT64)
    counted = g.op("CumSum", [separators, g.int64(1, "axis")])
    if spec.separator_opens:
        ids_segment = g.op("Max", [counted, g.int64(1, "one")])
    else:
        ids_segment = g.op("Add", [g.op("Sub", [counted, separators]), g.int64(1, "one")])
    segment = g.op("Where", [is_pad, g.int64(0, "zero"), ids_segment], base="segments")
    # a token starts its segment where its segment differs from the one before (the first always);
    # its position counts from the last start at or before it, a running max as numpy's accumulate
    before = g.op(
        "Concat",
        [g.op("Slice", [segment, g.int64([0]), g.int64([1]), g.int64([1])]), segment],
        axis=1,
    )
    previous = g.op("Slice", [before, g.int64([0]), g.int64([tokens]), g.int64([1])])
    first_column = np.zeros(tokens, dtype=bool)
    first_column[0] = True
    starts = g.op("Or", [g.op("Not", [g.op("Equal", [segment, previous])]), g.tensor(first_column, "first")])
    columns = np.arange(tokens, dtype=np.int64)
    start_columns = g.op("Where", [starts, g.tensor(columns, "columns"), g.int64(0, "zero")])
    # (N, T, T): start u's column where u <= t, else 0; its max over u is t's segment's first column
    reached = g.op(
        "Where",
        [
            g.tensor(columns[np.newaxis, :] <= columns[:, np.newaxis], "at_or_before"),
            g.op("Unsqueeze", [start_columns, g.int64([1])]),
            g.int64(0, "zero"),
        ],
    )
    first = g.op("ReduceMax", [reached, g.int64([2])], keepdims=0)
    positions = g.op("Sub", [g.tensor(columns, "columns"), first], base="positions")
    return Segments(segment, positions)


_INT64 = 7  # onnx.TensorProto.INT64
_BOOL = 9  # onnx.TensorProto.BOOL


def embedding(g: Graph, ids: str, name: str, table: Any) -> str:
    """Each id's row of the table: (N, T) int64 ids to (N, T, size)."""
    return g.op("Gather", [g.weight(name, table), ids], axis=0)


def position(g: Graph, x: str, name: str, table: Any, context: Segments | None) -> str:
    """x plus each token's position's row: by its position within its segment when the batch has them."""
    P = g.weight(name, table)
    if context is None:
        return g.op("Add", [x, P])
    return g.op("Add", [x, g.op("Gather", [P, context.positions], axis=0)])


def layer_norm(g: Graph, spec: LayerNorm, x: str, names: Sequence[str], weights: Sequence[Any]) -> str:
    """gamma (x - mean) / sqrt(var + epsilon) + beta over the last axis, as LayerNormArrayLayer."""
    gamma, beta = (g.weight(name, array) for name, array in zip(names, weights, strict=True))
    mean = g.op("ReduceMean", [x, g.int64([-1])], keepdims=1)
    centred = g.op("Sub", [x, mean])
    variance = g.op("ReduceMean", [g.op("Mul", [centred, centred]), g.int64([-1])], keepdims=1)
    deviation = g.op("Sqrt", [g.op("Add", [variance, g.constant(spec.epsilon)])])
    return g.op("Add", [g.op("Mul", [gamma, g.op("Div", [centred, deviation])]), beta])


def _affine(g: Graph, x: str, names: Sequence[str], weights: Sequence[Any]) -> str:
    # x W^T (+ b) over the last axis
    z = g.op("MatMul", [x, g.op("Transpose", [g.weight(names[0], weights[0])], perm=[1, 0])])
    return g.op("Add", [z, g.weight(names[1], weights[1])]) if len(weights) > 1 else z


def token_dense(g: Graph, spec: Dense, x: str, names: Sequence[str], weights: Sequence[Any]) -> str:
    """
    A token-wise dense layer: each token's W x + b, then the spec's activation, or a softmax over
    each token's classes for a token-wise output layer (TokenSoftmaxArrayLayer).
    """
    z = _affine(g, x, names, weights)
    if spec.output:
        return g.op("Softmax", [z], axis=-1)
    return activate(g, spec.activation, z)


def gated_dense(g: Graph, spec: GatedDense, x: str, names: Sequence[str], weights: Sequence[Any]) -> str:
    """The gated FFN: z = W x + b over (2 size), act(z[:size]) * z[size:], as GatedDenseArrayLayer."""
    z = _affine(g, x, names, weights)
    size = spec.size
    gate = g.op("Slice", [z, g.int64([0]), g.int64([size]), g.int64([-1])])
    value = g.op("Slice", [z, g.int64([size]), g.int64([2 * size]), g.int64([-1])])
    return g.op("Mul", [activate(g, spec.activation, gate), value])


def _heads(g: Graph, rows: str, tokens: int, heads: int, key_size: int) -> str:
    # (N, T, n * d_k) as (N, T, n, d_k)
    return g.op("Reshape", [rows, g.int64([-1, tokens, heads, key_size])])


def _rotated(g: Graph, x: str, cos: str, sin: str, half: int) -> str:
    # (N, T, n, d_k): each head's features j and j + d_k / 2 rotated by the token's angles
    a = g.op("Slice", [x, g.int64([0]), g.int64([half]), g.int64([-1])])
    b = g.op("Slice", [x, g.int64([half]), g.int64([2 * half]), g.int64([-1])])
    first = g.op("Sub", [g.op("Mul", [a, cos]), g.op("Mul", [b, sin])])
    second = g.op("Add", [g.op("Mul", [a, sin]), g.op("Mul", [b, cos])])
    return g.op("Concat", [first, second], axis=-1)


def allowed_scores(g: Graph, tokens: int, causal: bool, context: Segments | None, document_mask: bool) -> str | None:
    """
    The (N or 1, 1, T, T) scores a query may attend to, Masking's: causal, keys up to its own
    column; by segment, its own segment's keys but a pad's, its own column always. None for all.
    """
    columns = np.arange(tokens)
    allowed = g.tensor(columns[np.newaxis, :] <= columns[:, np.newaxis], "causal") if causal else None
    if context is not None:
        ids = context.segments
        if not document_mask:
            # attention across the paragraphs, pads still masked (the context-length workplan, D8)
            ids = g.op("Cast", [g.op("Not", [g.op("Equal", [ids, g.int64(0, "zero")])])], to=_INT64)
        queries = g.op("Unsqueeze", [ids, g.int64([2])])
        keys = g.op("Unsqueeze", [ids, g.int64([1])])
        same = g.op(
            "And",
            [g.op("Equal", [queries, keys]), g.op("Not", [g.op("Equal", [keys, g.int64(0, "zero")])])],
        )
        same = g.op("Or", [same, g.tensor(np.eye(tokens, dtype=bool), "own_column")])
        allowed = same if allowed is None else g.op("And", [same, allowed])
    if allowed is None:
        return None
    return g.op("Unsqueeze", [allowed, g.int64([-3])])


def _project(
    g: Graph, x: str, names: Sequence[str], weights: Sequence[Any], tokens: int, heads: int, key_size: int
) -> str:
    # x (N, T, d) through one of Q's, K's or V's (W, b) as (N, T, n, d_k)
    return _heads(g, _affine(g, x, names, weights), tokens, heads, key_size)


def _attend(
    g: Graph,
    queries: str,
    keys: str,
    values: str,
    heads: int,
    kv_heads: int,
    key_size: int,
    key_count: int | None,
    allowed: str | None,
    empty_rows: bool = False,
) -> str:
    """
    Attention's softmax over (N, h, T_q, d_k) queries and (N, g, T_k, d_k) keys and values: each
    head's softmax((Q K^T) / sqrt(d_k)) over its allowed keys times V, its key/value head
    i // (h / g); (N, h, T_q, d_k). key_count None when the keys' count is the graph's to find
    (a decode's cache). empty_rows: a query with no allowed key (cross-attention's pad) gives 0, as
    tiled_forward's H = 0, not softmax's NaN.
    """
    if kv_heads != heads:
        shared = heads // kv_heads
        keys, values = (
            g.op(
                "Reshape",
                [
                    g.op("Expand", [g.op("Unsqueeze", [t, g.int64([2])]), g.int64([1, kv_heads, shared, 1, 1])]),
                    g.int64([0, heads, -1, key_size]),
                ],
            )
            for t in (keys, values)
        )
    products = g.op("MatMul", [queries, g.op("Transpose", [keys], perm=[0, 1, 3, 2])])
    # divided by sqrt(d_k) as a vector, one per key: the same bits as the scalar, but onnxruntime's
    # MatMulScaleFusion folds a scalar into FusedMatMul's alpha, a float32 attribute, which costs a
    # float64 graph 3e-8 of every score
    root = float(np.sqrt(key_size))
    if key_count is None:
        count = g.op("Slice", [g.op("Shape", [keys]), g.int64([2]), g.int64([3]), g.int64([0])])
        scale = g.op("Expand", [g.constant(root, "root_key_size"), count])
    else:
        scale = g.constant(np.full(key_count, root), "scale")
    scores = g.op("Div", [products, scale])
    masked = g.constant(-np.inf, "minus_infinity")
    if empty_rows and allowed is not None:
        # a row with no allowed key masks with 0, not -inf, so its softmax is finite, then weighs 0
        reached = g.op(
            "Cast", [g.op("ReduceMax", [g.op("Cast", [allowed], to=_INT64), g.int64([-1])], keepdims=1)], to=_BOOL
        )
        masked = g.op("Where", [reached, masked, g.constant(0.0, "zero_score")])
    if allowed is not None:
        scores = g.op("Where", [allowed, scores, masked])
    weights = g.op("Softmax", [scores], axis=-1)
    if empty_rows and allowed is not None:
        weights = g.op("Where", [allowed, weights, g.constant(0.0, "zero_weight")])
    return g.op("MatMul", [weights, values])


def _combine(g: Graph, weighted: str, tokens: int, width: int, names: Sequence[str], weights: Sequence[Any]) -> str:
    # (N, h, T, d_k) side by side, (N, T, h d_k), through Wo
    side_by_side = g.op("Reshape", [g.op("Transpose", [weighted], perm=[0, 2, 1, 3]), g.int64([-1, tokens, width])])
    return _affine(g, side_by_side, names, weights)


def _split_parameters(names: Sequence[str], weights: Sequence[Any]) -> list[tuple[tuple[str, str], tuple[Any, Any]]]:
    # attention's eight parameters as (Wq, bq), (Wk, bk), (Wv, bv), (Wo, bo) pairs of names and arrays
    pairs = list(zip(names, weights, strict=True))
    return [((pairs[i][0], pairs[i + 1][0]), (pairs[i][1], pairs[i + 1][1])) for i in range(0, 8, 2)]


def attention(
    g: Graph,
    spec: Attention,
    x: str,
    tokens: int,
    features: int,
    names: Sequence[str],
    weights: Sequence[Any],
    context: Segments | None,
) -> str:
    """
    Multi-head attention over (N, T, d), AttentionArrayLayer's forward: Q, K, V projected (Q and K
    rotated for a rotary layer), each head's softmax((Q K^T) / sqrt(d_k)) over its allowed keys
    times V, its key/value head j = i // (h / g), the heads side by side through Wo. Its dropout
    is training's only (D12).
    """
    heads = spec.heads
    key_size = spec.head_size(features)
    kv_heads = spec.key_value_heads or heads
    q, k, v, o = _split_parameters(names, weights)
    queries = _project(g, x, *q, tokens, heads, key_size)
    keys = _project(g, x, *k, tokens, kv_heads, key_size)
    values = _project(g, x, *v, tokens, kv_heads, key_size)
    if spec.positions == "rotary":
        cos_table, sin_table = rotary_table(tokens, key_size, spec.rotary_base)
        cos = g.constant(np.array(cos_table)[:, np.newaxis, :], "rotary_cos")
        sin = g.constant(np.array(sin_table)[:, np.newaxis, :], "rotary_sin")
        queries, keys = _rotated(g, queries, cos, sin, key_size // 2), _rotated(g, keys, cos, sin, key_size // 2)
    # (N, n, T, d_k)
    queries, keys, values = (g.op("Transpose", [t], perm=[0, 2, 1, 3]) for t in (queries, keys, values))
    allowed = allowed_scores(g, tokens, spec.causal, context, spec.document_mask)
    weighted = _attend(g, queries, keys, values, heads, kv_heads, key_size, tokens, allowed)
    return _combine(g, weighted, tokens, heads * key_size, *o)


def cross_attention(
    g: Graph,
    spec: CrossAttention,
    x: str,
    tokens: int,
    features: int,
    memory: str,
    memory_tokens: int,
    names: Sequence[str],
    weights: Sequence[Any],
    query_segments: str | None,
    memory_segments: str | None,
) -> str:
    """
    A decoder's attention to the encoder's output, CrossAttentionArrayLayer's forward: Q from x, K
    and V from the memory (N, S, d_m); masked by segment alone, a query reading its own segment's
    keys but a pad's, never its own column; a query with no key (a pad's) gives H = 0, so bo.
    Never causal or rotary.
    """
    heads = spec.heads
    key_size = spec.head_size(features)
    kv_heads = spec.key_value_heads or heads
    q, k, v, o = _split_parameters(names, weights)
    queries = g.op("Transpose", [_project(g, x, *q, tokens, heads, key_size)], perm=[0, 2, 1, 3])
    keys = g.op("Transpose", [_project(g, memory, *k, memory_tokens, kv_heads, key_size)], perm=[0, 2, 1, 3])
    values = g.op("Transpose", [_project(g, memory, *v, memory_tokens, kv_heads, key_size)], perm=[0, 2, 1, 3])
    allowed = None
    if query_segments is not None and memory_segments is not None:
        same = g.op(
            "And",
            [
                g.op(
                    "Equal",
                    [
                        g.op("Unsqueeze", [query_segments, g.int64([2])]),
                        g.op("Unsqueeze", [memory_segments, g.int64([1])]),
                    ],
                ),
                g.op("Not", [g.op("Equal", [g.op("Unsqueeze", [memory_segments, g.int64([1])]), g.int64(0, "zero")])]),
            ],
        )
        allowed = g.op("Unsqueeze", [same, g.int64([-3])])
    weighted = _attend(g, queries, keys, values, heads, kv_heads, key_size, memory_tokens, allowed, empty_rows=True)
    return _combine(g, weighted, tokens, heads * key_size, *o)


@dataclass(frozen=True)
class StepCache:
    """
    A decode step's inputs for one self-attention layer, and its outputs: the past keys and values
    (N, g, L, d_k), rotated keys for a rotary layer, and their names in the graph's outputs.
    """

    past_keys: str
    past_values: str
    present_keys: str
    present_values: str


@dataclass(frozen=True)
class StepContext:
    """
    A decode step's own: the token's segment and position (N, 1) int64, its absolute column
    (a (1,) int64, rotary's), the past tokens' segments (N, L), and the cache's window, T.
    """

    segments: str
    positions: str
    column: str
    past_segments: str
    window: int


def attention_step(
    g: Graph,
    spec: Attention,
    x: str,
    features: int,
    names: Sequence[str],
    weights: Sequence[Any],
    cache: StepCache,
    step: StepContext,
) -> str:
    """
    One token a row, (N, 1, d), attending to its row's cache and itself (AttentionArrayLayer.step):
    its key and value appended, the oldest dropped past the window, the query masking another
    segment's keys and a pad's but its own, rotary by its column. The present keys and values are
    graph outputs, the next step's past.
    """
    heads = spec.heads
    key_size = spec.head_size(features)
    kv_heads = spec.key_value_heads or heads
    q, k, v, o = _split_parameters(names, weights)
    queries = _project(g, x, *q, 1, heads, key_size)
    keys = _project(g, x, *k, 1, kv_heads, key_size)
    values = _project(g, x, *v, 1, kv_heads, key_size)
    if spec.positions == "rotary":
        half = key_size // 2
        thetas = np.array([spec.rotary_base ** (-2 * j / key_size) for j in range(half)])
        angles = g.op("Mul", [g.op("Cast", [step.column], to=g.tensor_type), g.constant(thetas, "thetas")])
        cos = g.op("Reshape", [g.op("Cos", [angles]), g.int64([1, 1, half])])
        sin = g.op("Reshape", [g.op("Sin", [angles]), g.int64([1, 1, half])])
        queries, keys = _rotated(g, queries, cos, sin, half), _rotated(g, keys, cos, sin, half)
    queries, keys, values = (g.op("Transpose", [t], perm=[0, 2, 1, 3]) for t in (queries, keys, values))
    window = [g.int64([-step.window]), g.int64([2**62]), g.int64([2])]
    present_keys = g.op("Slice", [g.op("Concat", [cache.past_keys, keys], axis=2), *window])
    present_values = g.op("Slice", [g.op("Concat", [cache.past_values, values], axis=2), *window])
    g.rename(present_keys, cache.present_keys)
    g.rename(present_values, cache.present_values)
    segments = step.segments
    past = step.past_segments
    if not spec.document_mask:
        segments, past = (
            g.op("Cast", [g.op("Not", [g.op("Equal", [t, g.int64(0, "zero")])])], to=_INT64) for t in (segments, past)
        )
    key_segments = g.op(
        "Slice", [g.op("Concat", [past, segments], axis=1), g.int64([-step.window]), g.int64([2**62]), g.int64([1])]
    )
    # (N, 1, 1, c): its own segment's keys but a pad's, and its own column, the cache's last
    same = g.op(
        "And",
        [g.op("Equal", [key_segments, segments]), g.op("Not", [g.op("Equal", [key_segments, g.int64(0, "zero")])])],
    )
    count = g.op("Shape", [key_segments], start=1, end=2)
    last = g.op(
        "Equal",
        [
            g.op("Range", [g.int64(0, "zero"), g.op("Squeeze", [count]), g.int64(1, "one")]),
            g.op("Sub", [g.op("Squeeze", [count]), g.int64(1, "one")]),
        ],
    )
    allowed = g.op("Unsqueeze", [g.op("Or", [same, last]), g.int64([1, 2])])
    weighted = _attend(g, queries, cache.present_keys, cache.present_values, heads, kv_heads, key_size, None, allowed)
    return _combine(g, weighted, 1, heads * key_size, *o)
