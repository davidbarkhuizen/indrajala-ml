# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, Q, K, V, P, H, which strict mode takes for constants)
"""
Multi-head self-attention in pure Python (the multi-head attention workplan, D4, D9; README, Layer
norm and attention), the counterpart of attention_array_layer.py, in the same three blocks per
pass: project, attend (each head), combine. Every expression is attention_array_layer's, per
scalar, in its grouping: every product a left fold from 0.0 in index order (fold), and every
other row sum a left fold from its first value, numpy's cumsum's. One code path for every head
count.

Attend is tiled (the context-length workplan, D3): the keys in blocks of block (BLOCK, 16) cut at
the columns that are multiples of it (key_blocks), and per head, block and row reaching it
(Masking.rows), m' = max(m, the block's row max), e = exp(S - m') (0 in place of an all-masked
row's m'), l = l exp(m - m') + fold(e), H = H exp(m - m') + e V, then H / l. The backward pass
recomputes each block's P = exp(S - m) / l, takes D = fold(dH H) over d_k, dS = P (dP - D), dQ the
blocks' dS K summed in key order and dK = dS^T Q. Masked scores are -inf: causal (the sequence task
workplan, D7) and by segment (the segments workplan, D4), never a query's own column; a block no
query reaches is skipped, which, every sum a sequential fold, changes no bits.

The layer's per-example caches (Q, K, V, m, l, H, where its dropout draws are, its segments, and
backward dQ, dK, dV, dX) are its own, not its nodes': Q, K, V, H, dQ, dK and dV are matrices over
the tokens packed as the crate's, (T, h * d_k) (K, V, dK and dV (T, g * d_k), g the key/value
heads), head i's columns i * d_k to (i + 1) * d_k - 1, and
m and l h lists of T. It names them in example_fields, and the layer-major batch path
(layer_major.py) keeps them in lanes, as it does the nodes' fields. Every pass assigns new lists,
never writes into a cached one, so a lane can hold them without a copy.

A layer with dropout (the attention-dropout workplan, D3, D4, D6; the context-length workplan, D12)
drops its weights in training as numpy's does: per key block, a mask u >= dropout for every
example, head, query and key in that order, numpy's (N, h, T, b) draw, e dropped as e M / keep
before e V. Its network trains layer-major: begin_batch keeps the generator's state before the
batch's draws and moves the generator past all N h T T of them, and each example draws its own
(h, T, b) slab of each block from a copy advanced to it (Pcg64Generator.advance), forward and
again backward: dV = (P kept)^T dH and dP = dP kept, kept = M / keep. One example alone is a batch
of one. In inference, and at dropout 0, nothing is drawn.

A rotary layer (the rotary-positions workplan, D3, D4) ends project by rotating each head's Q and K
rows, token t's by rotary_table's row t, feature j < d_k / 2 paired with j + d_k / 2: q'_j = q_j c -
q_{j+d_k/2} s and q'_{j+d_k/2} = q_j s + q_{j+d_k/2} c, per scalar, numpy's expressions, so numpy's
bits on the same Q and K. The backward pass rotates dQ and dK back, dq_j = dq'_j c + dq'_{j+d_k/2} s
and dq_{j+d_k/2} = dq'_{j+d_k/2} c - dq'_j s, before project's products and the gradients. The
caches _Q and _K are the rotated ones, _dQ and _dK the unrotated ones' gradients.

A causal layer decodes (the generation workplan, D2, D3; the segments workplan, D8): prefill is the
forward pass's blocks on each of a batch's prompts' L <= T token rows, masked by segment when the
batch has segments, keeping each row's last segment's rotated keys and values in an AttentionCache;
step projects one token a row, rotates its query and key by its column's row of the table (extended
past T), appends its key and value to its row's cache, emptied first when the token starts a
segment, dropping the oldest past T, and attends from its query to every cached key through the
same tiles, the cache's blocks cut from its first key's column as the full pass's: the cache holds
the keys a masked pass would leave unmasked, and no others, so a decode's bits are the full pass's
within the window.

The softmax's exp is this module's exp, which tests may replace, as attention_array_layer's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import ClassVar

from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.python.batch_norm_layer import fold
from indrajala_ml.model.layers.python.residual_layer import ParameterFreeLayer
from indrajala_ml.model.layers.python.token_layer import (
    TokenNode,
    WeightRow,
    randomize_rows,
    restore_rows,
    snapshot_rows,
    token_values,
)
from indrajala_ml.model.protocols.decoding_protocols import refuse_non_causal_decoding
from indrajala_ml.model.protocols.layer_protocols import InputLayer, TrainableLayer
from indrajala_ml.model.specs.rotary_table import rotary_table
from indrajala_ml.pcg64 import Pcg64Generator, default_rng

exp = math.exp

Matrix = list[list[float]]


def _affine(X: Matrix, rows: Sequence[WeightRow]) -> Matrix:
    # X W^T + b: each token's product with each row, then the bias
    return [[fold([x_j * w_j for x_j, w_j in zip(x, row.weights)]) + row.bias for row in rows] for x in X]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return fold([a_i * b_i for a_i, b_i in zip(a, b)])


def _times_rows(A: Matrix, rows: Sequence[WeightRow]) -> Matrix:
    # A W, W's rows the weight rows: (A W)_tj = sum_k(A_tk * W_kj)
    return [[fold([a_k * row.weights[j] for a_k, row in zip(a, rows)]) for j in range(len(rows[0].weights))] for a in A]


# the tiled attend (the context-length workplan, D3, D12): attention_array_layer's, per example
BLOCK = 16


def key_blocks(keys: int, block: int, origin: int = 0) -> list[tuple[int, int]]:
    """attention_array_layer.key_blocks: bounds at the columns that are multiples of block, key 0 at origin."""
    first = (block - origin % block) % block or block
    bounds = [0, *range(first, keys, block), keys]
    return [(k0, k1) for k0, k1 in pairwise(bounds) if k0 < k1]


def _fold_first(values: Sequence[float]) -> float:
    # a left fold from the first value, numpy's cumsum's: ((x0 + x1) + x2) + ...
    total = values[0]
    for x in values[1:]:
        total += x
    return total


class Masking:
    """
    One example's masks (attention_array_layer.Masking's rules): causal, key u at most query t's
    column t + offset; and by segment, the queries' and the keys' ids, another segment's key or a
    pad's (0) masked, never the query's own column.
    """

    def __init__(
        self, causal: bool, offset: int = 0, segments: tuple[Sequence[int], Sequence[int]] | None = None
    ) -> None:
        self.causal, self.offset, self.segments = causal, offset, segments

    def allowed(self, t: int, u: int) -> bool:
        column = t + self.offset
        if self.causal and u > column:
            return False
        if self.segments is None:
            return True
        queries, keys = self.segments
        return u == column or (keys[u] == queries[t] and keys[u] != 0)

    def rows(self, queries: int, k0: int, k1: int) -> tuple[int, int] | None:
        """The rows reaching keys k0..k1, the smallest range holding them; None when none does."""
        r0 = max(0, k0 - self.offset) if self.causal else 0
        if r0 >= queries:
            return None
        if self.segments is None:
            return r0, queries
        reached = [t for t in range(r0, queries) if any(self.allowed(t, u) for u in range(k0, k1))]
        return (reached[0], reached[-1] + 1) if reached else None


Draws = list[list[list[float]]]


class AttentionCache:
    """
    One decode's rotated keys K and values V per row of its batch, rows of g * d_k: each row's current
    segment's, its last c <= T tokens', oldest first (the segments workplan, D8 (b)), and that
    segment's id (1 for every row of a batch without segments).
    """

    def __init__(self) -> None:
        self.K: list[Matrix] = []
        self.V: list[Matrix] = []
        self.segments: list[int] = []

    def append(self, n: int, k: Matrix, v: Matrix, segment: int, window: int) -> None:
        # row n's new key and value, after emptying its cache when they start a segment (or are a pad's)
        if segment != self.segments[n] or segment == 0:
            self.K[n], self.V[n], self.segments[n] = [], [], segment
        self.K[n] = (self.K[n] + k)[-window:]
        self.V[n] = (self.V[n] + v)[-window:]


class AttentionLayer(ParameterFreeLayer[TokenNode]):
    """
    Over tokens tokens of features features (d), in heads heads (h) of key_size features (d_k,
    d / h when None): Q = X Wq^T + bq, K and V likewise, each head's P[i] =
    softmax_rows((Q[i] K[j]^T) / sqrt(d_k)) and H[i] = P[i] V[j], j = i // (h / g) with g the
    key_value_heads (h when None; the grouped-query attention workplan, D2), out = H Wo^T + bo with
    H the heads side by side; a causal one masks S_ij for j > i, and a rotary one rotates Q's and
    K's heads by their tokens' positions. Its weight sets, also its draw and snapshot order, are
    Wq's h * d_k rows of d weights, then Wk's and Wv's g * d_k, then Wo's d rows of
    h * d_k weights, each row with its bias: the weights decayed and the biases not. Each projection
    is drawn as a dense layer, its rows' weights then its biases, Wq, Wk, Wv and Wo in turn. Hidden
    only, ending a token block's body.

    The backward pass runs whole in compute_hidden_deltas, since the gradients need dQ, dK and dV
    and the layer before reads dX.
    """

    example_fields: ClassVar[tuple[str, ...]] = (
        "_Q",
        "_K",
        "_V",
        "_m",
        "_l",
        "_H",
        "_draws",
        "_segments",
        "_dQ",
        "_dK",
        "_dV",
        "_dX",
    )

    def __init__(
        self,
        input_layer: InputLayer,
        tokens: int,
        features: int,
        heads: int = 1,
        key_size: int | None = None,
        causal: bool = False,
        dropout: float = 0.0,
        positions: str = "none",
        rotary_base: float = 10000.0,
        block: int = BLOCK,
        document_mask: bool = True,
        key_value_heads: int | None = None,
    ) -> None:
        assert len(input_layer.nodes) == tokens * features
        assert key_value_heads is None or (key_value_heads >= 1 and heads % key_value_heads == 0), (
            f"an attention's key/value heads divide its heads; got {key_value_heads} of {heads}"
        )
        assert block >= 1, f"an attention's key blocks hold at least one key; got {block}"
        # the key blocks' size (the context-length workplan, D3)
        self.block = block
        # False: attention across the paragraphs, pads still masked (the context-length workplan, D8)
        self.document_mask = document_mask
        assert 0.0 <= dropout < 1.0, f"an attention's dropout is in [0.0, 1.0); got {dropout}"
        self.input_layer = input_layer
        self.tokens = tokens
        self.features = features
        self.heads = heads
        self.key_size = features // heads if key_size is None else key_size
        self.width = heads * self.key_size
        # the key/value heads g (the grouped-query attention workplan, D2): query head i reads
        # key/value head i // (h / g), in the keys' and values' columns of that head
        self.key_value_heads = heads if key_value_heads is None else key_value_heads
        self.kv_width = self.key_value_heads * self.key_size
        self._group = heads // self.key_value_heads
        # computed once and divided by, never multiplied by its reciprocal
        self.scale = math.sqrt(self.key_size)
        self.causal = causal
        self.dropout = dropout
        self._keep = 1.0 - dropout
        # set by set_training_mode and set_rng, as a dense dropout layer's nodes'
        self.training = False
        self.rng: Pcg64Generator = default_rng()
        assert positions in ("none", "rotary"), f'an attention\'s positions are "none" or "rotary"; got {positions!r}'
        self.positions = positions
        self.rotary_base = rotary_base
        # rotary_table's cos and sin, (T, d_k / 2) each, or None unrotated
        self._table = rotary_table(tokens, self.key_size, rotary_base) if positions == "rotary" else None
        self.queries = [WeightRow(features) for _ in range(self.width)]
        self.keys, self.values = ([WeightRow(features) for _ in range(self.kv_width)] for _ in range(2))
        self.outputs = [WeightRow(self.width) for _ in range(features)]
        self.nodes = [TokenNode() for _ in range(tokens * features)]
        self.size = len(self.nodes)

        # this example's caches, set by the forward and backward passes: m and l per head and token
        # (h lists of T), never the weights
        self._Q: Matrix
        self._K: Matrix
        self._V: Matrix
        self._m: Matrix
        self._l: Matrix
        # where this example's dropout draws are (the generator's state before its batch's draws,
        # the batch's examples and this example's index: D12), or None when it drew none
        self._draws: tuple[dict[str, object], int, int] | None = None
        self._segments: list[int] | None = None
        # a batch's generator state and size while the layer-major pass runs it (begin_batch)
        self._batch: tuple[dict[str, object], int] | None = None
        self._example = 0
        self._H: Matrix
        self._dQ: Matrix
        self._dK: Matrix
        self._dV: Matrix
        self._dX: list[float]

    @property
    def draws(self) -> bool:
        """Whether a training pass draws from the generator: at dropout > 0 (the network's _layer_major)."""
        return self.dropout > 0.0

    def set_rng(self, rng: Pcg64Generator) -> None:
        self.rng = rng

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def _inputs(self) -> Matrix:
        return [token_values(self.input_layer.nodes, t, self.features) for t in range(self.tokens)]

    def _deltas(self) -> Matrix:
        d = self.features
        return [[node.delta for node in self.nodes[t * d : (t + 1) * d]] for t in range(self.tokens)]

    def _angles(self, start: int, tokens: int) -> tuple[Matrix, Matrix]:
        # the table's rows for positions start to start + tokens - 1, extended (doubled) past its rows,
        # which don't change
        assert self._table is not None
        if start + tokens > len(self._table[0]):
            positions = max(start + tokens, 2 * len(self._table[0]))
            self._table = rotary_table(positions, self.key_size, self.rotary_base)
        cos, sin = self._table
        return cos[start : start + tokens], sin[start : start + tokens]

    def _rotated(self, A: Matrix, inverse: bool = False, start: int = 0) -> Matrix:
        # (T, n * d_k), n query or key/value heads, each head's features j and j + d_k / 2 of row t
        # rotated by token t's angles, or back; a decode's rows from position start
        d_k, half = self.key_size, self.key_size // 2
        out: Matrix = []
        for row, cos, sin in zip(A, *self._angles(start, len(A)), strict=True):
            new = list(row)
            for i in range(len(row) // d_k):
                for j in range(half):
                    a, b, c, s = row[i * d_k + j], row[i * d_k + j + half], cos[j], sin[j]
                    if inverse:
                        new[i * d_k + j], new[i * d_k + j + half] = a * c + b * s, b * c - a * s
                    else:
                        new[i * d_k + j], new[i * d_k + j + half] = a * c - b * s, a * s + b * c
            out.append(new)
        return out

    # project, attend, combine: the forward pass's blocks

    def _project(self, X: Matrix, start: int = 0) -> tuple[Matrix, Matrix, Matrix]:
        # each over all heads, (T, h * d_k); a rotary layer's Q and K then rotated (a decode's rows
        # from position start)
        Q, K, V = _affine(X, self.queries), _affine(X, self.keys), _affine(X, self.values)
        if self._table is not None:
            return self._rotated(Q, start=start), self._rotated(K, start=start), V
        return Q, K, V

    def _block_draws(self, draws: tuple[dict[str, object], int, int], b: int, k0: int) -> Draws:
        # this example's (h, T, b) masks of the key block from k0, numpy's (N, h, T, b) draw's slab:
        # the batch's earlier blocks' N h T b' values, then the earlier examples' h T b, skipped by
        # advancing a copy of the batch's generator (D12)
        state, examples, n = draws
        T = self.tokens
        offset = examples * self.heads * T * k0 + n * self.heads * T * b
        rng = default_rng()
        rng.state = dict(state)
        rng.advance(offset)
        return [
            [[1.0 if rng.random() >= self.dropout else 0.0 for _ in range(b)] for _ in range(T)]
            for _ in range(self.heads)
        ]

    def _attention_segments(self, segments: Sequence[Sequence[int]] | None, n: int) -> list[int] | None:
        # row n's segment ids attention masks by: the batch's, or with document_mask False only the
        # pads' (0), every other token one segment
        if segments is None:
            return None
        return [int(s) for s in segments[n]] if self.document_mask else [0 if s == 0 else 1 for s in segments[n]]

    def begin_batch(self, examples: int) -> None:
        """
        The layer-major pass's batch of examples: the generator's state before its draws, which each
        example draws its own masks from, and the generator moved past all N h T T of them (D12).
        """
        if self.training and self.dropout > 0.0:
            self._batch = (self.rng.state, examples)
            self.rng.advance(examples * self.heads * self.tokens * self.tokens)

    def begin_example(self, example: int) -> None:
        self._example = example

    def end_batch(self) -> None:
        self._batch = None

    def _tiled(
        self,
        QKV: tuple[Matrix, Matrix, Matrix],
        masking: Masking,
        origin: int = 0,
        draws: tuple[dict[str, object], int, int] | None = None,
    ) -> tuple[Matrix, Matrix, Matrix]:
        # H (T_q, h * d_k), m and l (h lists of T_q): per key block in order, per head, the rows
        # reaching it, m' = max(m, the block's row max), e = exp(S - m') (0 in place of an
        # all-masked row's m'), l = l exp(m - m') + fold(e), H = H exp(m - m') + e V (e dropped as e
        # M / keep), then H / l: attention_array_layer.tiled_forward's expressions per scalar
        Q, K, V = QKV
        queries, d_k = len(Q), self.key_size
        m = [[-math.inf] * queries for _ in range(self.heads)]
        l = [[0.0] * queries for _ in range(self.heads)]
        H = [[0.0] * self.width for _ in range(queries)]
        for k0, k1 in key_blocks(len(K), self.block, origin):
            masks = None if draws is None else self._block_draws(draws, k1 - k0, k0)
            rows = masking.rows(queries, k0, k1)
            if rows is None:
                continue
            for i in range(self.heads):
                c, ck = i * d_k, i // self._group * d_k
                for t in range(*rows):
                    s = [
                        _dot(Q[t][c : c + d_k], K[u][ck : ck + d_k]) / self.scale
                        if masking.allowed(t, u)
                        else -math.inf
                        for u in range(k0, k1)
                    ]
                    m_old = m[i][t]
                    m_new = max(m_old, *s)
                    shift = 0.0 if m_new == -math.inf else m_new
                    e = [exp(x - shift) for x in s]
                    alpha = exp(m_old - shift)
                    l[i][t] = l[i][t] * alpha + _fold_first(e)
                    m[i][t] = m_new
                    if masks is not None:
                        e = [x * kept / self._keep for x, kept in zip(e, masks[i][t])]
                    for j in range(d_k):
                        ev = fold([e_u * V[k0 + u][ck + j] for u, e_u in enumerate(e)])
                        H[t][c + j] = H[t][c + j] * alpha + ev
        for t in range(queries):
            for i in range(self.heads):
                for j in range(i * d_k, (i + 1) * d_k):
                    H[t][j] /= l[i][t]
        return H, m, l

    def _combine(self, H: Matrix) -> Matrix:
        # one product over all heads' columns
        return _affine(H, self.outputs)

    def forward(self, *, context: ForwardContext = NO_CONTEXT) -> None:
        self._Q, self._K, self._V = self._project(self._inputs())
        self._segments = self._attention_segments(context.segments, 0)
        masking = Masking(self.causal, 0, None if self._segments is None else (self._segments, self._segments))
        self._draws = None
        if self.training and self.dropout > 0.0:
            if self._batch is None:
                # one example alone: a batch of one, its draws taken now
                self._draws = (self.rng.state, 1, 0)
                self.rng.advance(self.heads * self.tokens * self.tokens)
            else:
                self._draws = (self._batch[0], self._batch[1], self._example)
        self._H, self._m, self._l = self._tiled((self._Q, self._K, self._V), masking, draws=self._draws)
        out = self._combine(self._H)
        for node, value in zip(self.nodes, (value for row in out for value in row)):
            node.activate(value)

    # decoding (the generation workplan, D2, D3)

    def start_decoding(self) -> AttentionCache:
        refuse_non_causal_decoding(self.causal)
        return AttentionCache()

    def prefill(self, X: Matrix, cache: AttentionCache, context: ForwardContext) -> Matrix:
        # N prompts' N * L rows, L <= T, at columns 0 to L - 1: the forward pass's blocks on each
        # row's L tokens, masked by segment when the batch has segments; each row keeps its last
        # segment's keys and values
        L = len(X) // context.examples
        out: Matrix = []
        for n in range(context.examples):
            Q, K, V = self._project(X[n * L : (n + 1) * L])
            segments = self._attention_segments(context.segments, n)
            masking = Masking(self.causal, 0, None if segments is None else (segments, segments))
            H, _m, _l = self._tiled((Q, K, V), masking)
            last = 1 if segments is None else segments[-1]
            keep = [t for t in range(L) if segments is None or segments[t] == last]
            cache.K.append([K[t] for t in keep][-self.tokens :])
            cache.V.append([V[t] for t in keep][-self.tokens :])
            cache.segments.append(last)
            out += self._combine(H)
        return out

    def step(self, x: Matrix, cache: AttentionCache, context: ForwardContext) -> Matrix:
        # one token a row at context.column: each query against its row's cache, its own key and
        # value appended, unmasked, since the cache holds its segment's keys only (D8 (b))
        out: Matrix = []
        for n, row in enumerate(x):
            q, k, v = self._project([row], start=context.column)
            attended = self._attention_segments(context.segments, n)
            segment = 1 if attended is None else attended[0]
            cache.append(n, k, v, segment, self.tokens)
            # the query is the cache's last column; the cache's first key is at origin, so its key
            # blocks are cut where the full pass's are (the context-length workplan, D3)
            c = len(cache.K[n])
            masking = Masking(True, c - 1)
            H, _m, _l = self._tiled((q, cache.K[n], cache.V[n]), masking, origin=context.column + 1 - c)
            out += self._combine(H)
        return out

    # the backward pass's blocks, last first

    def _combine_backward(self, delta: Matrix) -> Matrix:
        # dH = delta Wo, (T, h * d_k)
        return _times_rows(delta, self.outputs)

    def _attend_backward(self, dH: Matrix) -> tuple[Matrix, Matrix, Matrix]:
        # dQ, dK and dV, packed (T, h * d_k): per key block in order (the same masks drawn again),
        # per head, the rows reaching it, P = exp(S - m) / l recomputed, dP = dH V^T, with kept = M /
        # keep dV = (P kept)^T dH and dP = dP kept, dS = P (dP - D), D = fold(dH H) over d_k, dQ the
        # blocks' dS K summed in key order, dK = dS^T Q; dK and dV (T, g * d_k) per key block a
        # group's first head's assigned and each later head's added, in head order (D3); dQ and dK
        # then divided by the scale
        Q, K, V, T, d_k = self._Q, self._K, self._V, self.tokens, self.key_size
        segments = self._segments
        masking = Masking(self.causal, 0, None if segments is None else (segments, segments))
        D = [
            [_fold_first([dH[t][j] * self._H[t][j] for j in range(i * d_k, (i + 1) * d_k)]) for t in range(T)]
            for i in range(self.heads)
        ]
        dQ = [[0.0] * self.width for _ in range(T)]
        dK = [[0.0] * self.kv_width for _ in range(T)]
        dV = [[0.0] * self.kv_width for _ in range(T)]
        for k0, k1 in key_blocks(T, self.block):
            masks = None if self._draws is None else self._block_draws(self._draws, k1 - k0, k0)
            rows = masking.rows(T, k0, k1)
            if rows is None:
                continue
            r0, r1 = rows
            for i in range(self.heads):
                c, ck, first = i * d_k, i // self._group * d_k, i % self._group == 0
                P: Matrix = []
                dS: Matrix = []
                kept_rows: Matrix = []
                for t in range(r0, r1):
                    s = [
                        _dot(Q[t][c : c + d_k], K[u][ck : ck + d_k]) / self.scale
                        if masking.allowed(t, u)
                        else -math.inf
                        for u in range(k0, k1)
                    ]
                    p = [exp(x - self._m[i][t]) / self._l[i][t] for x in s]
                    dp = [_dot(dH[t][c : c + d_k], V[u][ck : ck + d_k]) for u in range(k0, k1)]
                    kept = [1.0] * len(p) if masks is None else [x / self._keep for x in masks[i][t]]
                    if masks is not None:
                        dp = [x * k for x, k in zip(dp, kept)]
                    P.append(p)
                    kept_rows.append(kept)
                    dS.append([p_u * (dp_u - D[i][t]) for p_u, dp_u in zip(p, dp)])
                for t, ds in zip(range(r0, r1), dS):
                    for j in range(d_k):
                        dQ[t][c + j] += fold([ds_u * K[k0 + u][ck + j] for u, ds_u in enumerate(ds)])
                for u in range(k1 - k0):
                    for j in range(d_k):
                        dk = fold([ds[u] * Q[r0 + r][c + j] for r, ds in enumerate(dS)])
                        if masks is None:
                            dv = fold([p[u] * dH[r0 + r][c + j] for r, p in enumerate(P)])
                        else:
                            dv = fold(
                                [(p[u] * kept[u]) * dH[r0 + r][c + j] for r, (p, kept) in enumerate(zip(P, kept_rows))]
                            )
                        if first:
                            dK[k0 + u][ck + j], dV[k0 + u][ck + j] = dk, dv
                        else:
                            dK[k0 + u][ck + j] += dk
                            dV[k0 + u][ck + j] += dv
        return (
            [[x / self.scale for x in row] for row in dQ],
            [[x / self.scale for x in row] for row in dK],
            dV,
        )

    def _project_backward(self, dQ: Matrix, dK: Matrix, dV: Matrix) -> list[float]:
        # three products over all heads, summed in this order
        dXq, dXk, dXv = _times_rows(dQ, self.queries), _times_rows(dK, self.keys), _times_rows(dV, self.values)
        return [(q + k) + v for q_row, k_row, v_row in zip(dXq, dXk, dXv) for q, k, v in zip(q_row, k_row, v_row)]

    def compute_hidden_deltas(self, next_layer: TrainableLayer, *, context: ForwardContext = NO_CONTEXT) -> None:
        for own_index, node in enumerate(self.nodes):
            node.delta = next_layer.downstream_sum(own_index)
        dH = self._combine_backward(self._deltas())
        self._dQ, self._dK, self._dV = self._attend_backward(dH)
        if self._table is not None:
            # project's backward starts by rotating back: the gradients of the unrotated Q and K
            self._dQ, self._dK = self._rotated(self._dQ, inverse=True), self._rotated(self._dK, inverse=True)
        self._dX = self._project_backward(self._dQ, self._dK, self._dV)

    def downstream_sum(self, own_index: int) -> float:
        return self._dX[own_index]

    def accumulate_gradients(self, *, context: ForwardContext = NO_CONTEXT) -> None:
        # each projection's rows over this example's tokens in order, after the examples before it
        X = self._inputs()
        for rows, gradient, inputs in (
            (self.queries, self._dQ, X),
            (self.keys, self._dK, X),
            (self.values, self._dV, X),
            (self.outputs, self._deltas(), self._H),
        ):
            for g_row, x in zip(gradient, inputs):
                for row, g in zip(rows, g_row):
                    accum = row.weight_gradient_accum
                    for j, x_j in enumerate(x):
                        accum[j] += g * x_j
                    row.bias_gradient_accum += g

    def weight_sets(self) -> Sequence[WeightRow]:
        return [*self.queries, *self.keys, *self.values, *self.outputs]

    def randomize_fan_in_aware(self, rng: Pcg64Generator) -> None:
        # each projection as a matrix, q, k, v, o in turn: numpy's order
        for rows in (self.queries, self.keys, self.values, self.outputs):
            randomize_rows(rows, rng)

    def snapshot_state(self) -> list[tuple[list[float], float]]:
        return snapshot_rows(self.weight_sets())

    def restore_state(self, layer_snapshot: Sequence[tuple[list[float], float]]) -> None:
        restore_rows(self.weight_sets(), layer_snapshot)
