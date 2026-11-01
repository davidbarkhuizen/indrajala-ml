# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, Q, K, V, P, H, which strict mode takes for constants)
"""
Multi-head self-attention in numpy (the multi-head attention workplan, D4; README, Layer norm and
attention): every expression the README's, in its grouping, in its three blocks per pass. project
is the products on the (N * T, d) rows, giving (N * T, h * d_k); attend works on the
(N, h, T, d_k) view of those rows, its products np.matmul over the (N, h) leading axes; combine
puts the heads side by side again and projects. One code path for every head count: a one-head
layer is the same blocks with h = 1. Every sum but a product is a left fold through np.cumsum
along the last axis (tests/model/layers/test_summation_order.py).

Attend is tiled (the context-length workplan, D3): the keys in blocks of block (BLOCK, 16), and per
query row a running max m, sum l and unnormalized output H; for each key block in order, m' =
max(m, the block's row max), e = exp(S - m'), l = l exp(m - m') + the fold of e, H = H exp(m - m')
+ e V, and at the end H / l (tiled_forward). The forward keeps m and l, (N, h, T, 1), never the
(N, h, T, T) weights; the backward recomputes each block's P = exp(S - m) / l and takes D = the fold
of dH H over d_k (FlashAttention-2's form of the softmax's row term), dS = P (dP - D), dQ the
blocks' dS K summed in key order, dK and dV per block (tiled_backward). Memory is linear in T.

The masks are built per block (Masking): causal, key u masked for query t when u > t (the sequence
task workplan, D7); by segment, another segment's key or a pad's, never the query's own (the
segments workplan, D4); and a band, for the decode tests' sliding reference. A masked score is
-inf, so its weight is exactly 0; a block no query may attend to is skipped, and a row wholly
masked within a computed block shifts by 0 in place of its -inf max, so its e is 0 and its l and H
unchanged. Each block's rows are the range of queries that reach it (causal: from its first key).

A layer with dropout (the attention-dropout workplan, D3, D4; the context-length workplan, D12)
drops its weights in training: per key block, an (N, h, T, b) mask M drawn from the network's
generator, every entry drawn, the masked ones included, M = u >= dropout, so a forward batch draws
N h T T uniforms as before, key block by key block; e is dropped as e M / keep before e V. The
generator's state before the draws is kept, and the backward pass draws the same masks again
from a copy: dV = (P M / keep)^T dH and dP = (dH V^T) M / keep. In inference, and at dropout 0,
nothing is drawn.

A rotary layer (the rotary-positions workplan, D3, D4) ends project by rotating each head's Q and K,
after their biases, by their token's position t: feature j < d_k / 2 paired with j + d_k / 2,
q'_j = q_j c - q_{j+d_k/2} s and q'_{j+d_k/2} = q_j s + q_{j+d_k/2} c, with c and s from
rotary_table, read on the (N, T, h, d_k) view of the rows. Attend reads the rotated Q and K; the
backward pass rotates dQ and dK back, dq_j = dq'_j c + dq'_{j+d_k/2} s and dq_{j+d_k/2} =
dq'_{j+d_k/2} c - dq'_j s, before project's products and the gradients. V isn't rotated. An
unrotated layer computes what it did before the rotation.

A causal layer decodes (the generation workplan, D2, D3; the segments workplan, D8): prefill is the
forward pass's blocks on a batch of N prompts' L <= T tokens, masked by segment when the batch has
segments, and keeps their rotated keys and values, (N, L, h * d_k) each, and their segments, in an
AttentionCache; step projects one token a row, rotates its query and key by its column
(rotary_table extended past T), appends its key and value to its row's cache, dropping the oldest
past T, and attends from its query to every cached key of its row through the same tiled attend
(the query the cache's last column), masking another segment's keys and pads' (D4). So a rotary
layer's window slides exactly: scores depend on distances up to T - 1, as in training.

The softmax's exp is this module's exp, which tests replace with another implementation's (it
isn't correctly rounded), as batch norm's tests replace sigmoid.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt

from indrajala_ml.model.layers.array.array_parameters import AttentionProjections
from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.batch_norm_array_layer import sum_rows
from indrajala_ml.model.layers.numpy.token_array_layer import BatchShaped
from indrajala_ml.model.protocols.decoding_protocols import refuse_non_causal_decoding
from indrajala_ml.model.specs.rotary_table import rotary_table

exp = np.exp


def _transpose(stack: FloatArray) -> FloatArray:
    # each matrix of a stack transposed
    return stack.swapaxes(-1, -2)


# the tiled attention (the context-length workplan, D3, D12): the keys in blocks of BLOCK, an online
# softmax per query row, nothing (N, h, T, T) kept
BLOCK = 16

Segments = npt.NDArray[np.intp]
Draw = Callable[[tuple[int, ...]], FloatArray]


@dataclass(frozen=True)
class Masking:
    """
    Which keys a query may attend to: causal, key u at most query t's column t + offset (a decode's
    queries are the keys' last, offset = keys - queries); band, when set, only the band keys before
    and including that column (a sliding window); and by segment, the queries' (N, T_q) and the
    keys' (N, T_k) ids, another segment's or a pad's key masked (the segments workplan, D4), never
    the query's own column.
    """

    causal: bool
    offset: int = 0
    band: int | None = None
    query_segments: Segments | None = None
    key_segments: Segments | None = None

    def tile(self, queries: int, k0: int, k1: int) -> tuple[int, int, npt.NDArray[np.bool_] | None] | None:
        """
        Key block k0:k1's rows r0:r1, the smallest range holding every query that may attend to
        one of its keys in some example, and its (N or 1, 1, r1 - r0, k1 - k0) mask of allowed
        scores, None when every score is allowed; None for a block no query may attend to.
        """
        r0, r1 = 0, queries
        if self.causal:
            r0 = max(0, k0 - self.offset)
        if self.band is not None:
            r1 = min(r1, k1 - 1 - self.offset + self.band)
        if r0 >= r1:
            return None
        rows = np.arange(r0, r1)[:, np.newaxis] + self.offset
        keys = np.arange(k0, k1)[np.newaxis, :]
        allowed: npt.NDArray[np.bool_] | None = None
        if self.causal and k1 - 1 > r0 + self.offset:
            allowed = keys <= rows
        if self.band is not None and k0 <= r1 - 1 + self.offset - self.band:
            inside = keys > rows - self.band
            allowed = inside if allowed is None else allowed & inside
        if self.query_segments is not None:
            assert self.key_segments is not None
            q = self.query_segments[:, r0:r1, np.newaxis]
            k = self.key_segments[:, np.newaxis, k0:k1]
            same: npt.NDArray[np.bool_] = ((q == k) & (k != 0)) | (keys == rows)
            joint = same if allowed is None else same & allowed
            reached = joint.any(axis=(0, 2))
            if not reached.any():
                return None
            first, last = int(np.argmax(reached)), len(reached) - int(np.argmax(reached[::-1]))
            return r0 + first, r0 + last, joint[:, np.newaxis, first:last]
        return r0, r1, allowed


def key_blocks(keys: int, block: int, origin: int = 0) -> list[tuple[int, int]]:
    """
    The key blocks k0:k1 of keys keys whose first is at column origin: their bounds at the columns
    that are multiples of block, so a decode's cache (from the first column it holds) is cut where
    the full pass over every column is, and its online softmax is the full pass's, block for block.
    """
    bounds = [0, *range((block - origin % block) % block or block, keys, block), keys]
    return [(k0, k1) for k0, k1 in pairwise(bounds) if k0 < k1]


def _scores(Q: FloatArray, K: FloatArray, scale: float, mask: npt.NDArray[np.bool_] | None) -> FloatArray:
    S = (Q @ _transpose(K)) / scale
    return S if mask is None else np.where(mask, S, -np.inf)


def tiled_forward(
    Q: FloatArray,
    K: FloatArray,
    V: FloatArray,
    scale: float,
    masking: Masking,
    block: int,
    draw: Draw | None,
    keep: float,
    origin: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    H, (N, h, T_q, d_k), and the rows' max m and sum l, (N, h, T_q, 1), from Q (N, h, T_q, d_k) and
    K and V (N, h, T_k, d_k): per key block, in order, m' = max(m, the block's row max), e = exp(S -
    m') (0 in place of an all-masked row's m'), l = l exp(m - m') + the fold of e, H = H exp(m - m')
    + e V (e dropped: e M / keep, M drawn by draw for the block's every query and key), then H / l.
    The key blocks are key_blocks(T_k, block, origin): origin is key 0's column.
    """
    N, h, queries, d_k = Q.shape
    keys = K.shape[2]
    m = np.full((N, h, queries, 1), -np.inf)
    l = np.zeros((N, h, queries, 1))
    H = np.zeros((N, h, queries, d_k))
    for k0, k1 in key_blocks(keys, block, origin):
        M = None if draw is None else draw((N, h, queries, k1 - k0))
        tile = masking.tile(queries, k0, k1)
        if tile is None:
            continue
        r0, r1, mask = tile
        S = _scores(Q[:, :, r0:r1], K[:, :, k0:k1], scale, mask)
        m_old = m[:, :, r0:r1]
        m_new = np.maximum(m_old, S.max(axis=-1, keepdims=True))
        shift = np.where(m_new == -np.inf, 0.0, m_new)
        e = exp(S - shift)
        alpha = exp(m_old - shift)
        l[:, :, r0:r1] = l[:, :, r0:r1] * alpha + np.cumsum(e, axis=-1)[..., -1:]
        dropped = e if M is None else e * M[:, :, r0:r1] / keep
        H[:, :, r0:r1] = H[:, :, r0:r1] * alpha + dropped @ V[:, :, k0:k1]
        m[:, :, r0:r1] = m_new
    return H / l, m, l


def tiled_backward(
    dH: FloatArray,
    saved: tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray],
    scale: float,
    masking: Masking,
    block: int,
    draw: Draw | None,
    keep: float,
    origin: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    dQ, dK and dV from dH and the forward's Q, K, V, H, m and l: per key block, P = exp(S - m) / l
    recomputed (M drawn again, the same draws in the same order), dV = (P M / keep)^T dH, dP = (dH
    V^T) M / keep, dS = P (dP - D) with D = the fold of dH H over d_k (FlashAttention-2's), dQ the
    blocks' dS K summed in key order, dK = dS^T Q; dQ and dK then divided by the scale.
    """
    Q, K, V, H, m, l = saved
    N, h, queries, _ = Q.shape
    keys = K.shape[2]
    D = np.cumsum(dH * H, axis=-1)[..., -1:]
    dQ, dK, dV = np.zeros_like(Q), np.zeros_like(K), np.zeros_like(V)
    for k0, k1 in key_blocks(keys, block, origin):
        M = None if draw is None else draw((N, h, queries, k1 - k0))
        tile = masking.tile(queries, k0, k1)
        if tile is None:
            continue
        r0, r1, mask = tile
        S = _scores(Q[:, :, r0:r1], K[:, :, k0:k1], scale, mask)
        P = exp(S - m[:, :, r0:r1]) / l[:, :, r0:r1]
        rows = dH[:, :, r0:r1]
        dP = rows @ _transpose(V[:, :, k0:k1])
        if M is None:
            dV[:, :, k0:k1] = _transpose(P) @ rows
        else:
            kept = M[:, :, r0:r1] / keep
            dV[:, :, k0:k1] = _transpose(P * kept) @ rows
            dP = dP * kept
        dS = P * (dP - D[:, :, r0:r1])
        dQ[:, :, r0:r1] += dS @ K[:, :, k0:k1]
        dK[:, :, k0:k1] = _transpose(dS) @ Q[:, :, r0:r1]
    return dQ / scale, dK / scale, dV


def _generator_at(state: Mapping[str, Any]) -> np.random.Generator:
    # a fresh generator in a saved state: the dropout draws again in the backward pass (D12)
    bit_generator = getattr(np.random, state["bit_generator"])()
    bit_generator.state = state
    return np.random.Generator(bit_generator)


class AttentionCache:
    """
    One decode's rotated keys K and values V, (N, c, h * d_k) each, the last c <= T columns' of the
    batch's N rows, oldest first; and each key's segment, (N, c), or None while the batch has none
    (every key then in segment 1).
    """

    def __init__(self, width: int) -> None:
        self.width = width
        self.K: FloatArray = np.zeros((0, 0, width))
        self.V: FloatArray = np.zeros((0, 0, width))
        self.segments: npt.NDArray[np.intp] | None = None

    def start(self, K: FloatArray, V: FloatArray, segments: npt.NDArray[np.intp] | None) -> None:
        self.K, self.V, self.segments = K, V, segments

    def append(self, K: FloatArray, V: FloatArray, segments: npt.NDArray[np.intp] | None, window: int) -> None:
        # K and V (N, 1, h * d_k), segments (N, 1) or None
        if segments is not None and self.segments is None:
            self.segments = np.ones(self.K.shape[:2], dtype=np.intp)
        if self.segments is not None:
            assert segments is not None, "a batch with segments keeps them"
            self.segments = np.concatenate((self.segments, segments), axis=1)[:, -window:]
        self.K = np.concatenate((self.K, K), axis=1)[:, -window:]
        self.V = np.concatenate((self.V, V), axis=1)[:, -window:]


class AttentionArrayLayer(BatchShaped, AttentionProjections[FloatArray]):
    """
    Per example over its tokens tokens of features features (d), in heads heads (h) of key_size
    features (d_k, d / h when None): Q = X Wq^T + bq, K and V likewise, each head's
    P[i] = softmax_rows((Q[i] K[i]^T) / sqrt(d_k)) and H[i] = P[i] V[i], out = H Wo^T + bo with H
    the heads side by side; a causal one masks S_ij for j > i, and one with dropout drops P's
    entries in training, and a rotary one rotates Q's and K's heads by their tokens' positions.
    Its parameters, also its draw order,
    are Wq, bq, Wk, bk, Wv, bv, Wo, bo: Wq, Wk, Wv (h * d_k, d) and Wo (d, h * d_k), heads as row
    (Wo: column) blocks, each drawn as a dense layer's W then b, the weights decayed and the biases
    not. Hidden only, ending a token block's body.

    The backward pass runs whole in _backward, since the gradients need dQ, dK and dV and the
    layer before reads dX.
    """

    def __init__(
        self,
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
    ) -> None:
        self._set_up(tokens, features, heads, key_size)
        # False: attention across the paragraphs, pads still masked (the context-length workplan, D8)
        self.document_mask = document_mask
        # computed once and divided by, never multiplied by its reciprocal
        self.scale = math.sqrt(self.key_size)
        self.causal = causal
        # the key blocks' size (D3): a layer option, not a spec field, since it groups the sums, not the model
        assert block >= 1, f"an attention's key blocks hold at least one key; got {block}"
        self.block = block
        # a band of keys, the last band columns up to each query's, when set (a sliding window: the
        # decode tests' reference); None, every earlier key
        self._band: int | None = None
        assert 0.0 <= dropout < 1.0, f"an attention's dropout is in [0.0, 1.0); got {dropout}"
        self.dropout = dropout
        self._keep = 1.0 - dropout
        # set by set_training_mode and set_rng, as a dense dropout layer's; the forward pass's mask,
        # None when it drew none, is what the backward pass reads
        self.training = False
        self.rng = np.random.default_rng()
        # the dropout generator's state before the forward pass's draws, which the backward pass
        # draws again (D12); None when the forward drew none
        self._dropout_state: Mapping[str, Any] | None = None
        assert positions in ("none", "rotary"), f'an attention\'s positions are "none" or "rotary"; got {positions!r}'
        self.positions = positions
        self.rotary_base = rotary_base
        self._rotary = positions == "rotary"
        if self._rotary:
            self._extend_table(tokens)

    def _extend_table(self, positions: int) -> None:
        cos, sin = rotary_table(positions, self.key_size, self.rotary_base)
        # (positions, 1, d_k / 2): each token's row, the same for every example and head
        self._cos = np.array(cos)[:, np.newaxis, :]
        self._sin = np.array(sin)[:, np.newaxis, :]

    def _angles(self, start: int, tokens: int) -> tuple[FloatArray, FloatArray]:
        # cos and sin of positions start to start + tokens - 1, the table extended (doubled) past its rows; the
        # rows already there don't change
        if start + tokens > self._cos.shape[0]:
            self._extend_table(max(start + tokens, 2 * self._cos.shape[0]))
        return self._cos[start : start + tokens], self._sin[start : start + tokens]

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def _zeros(self) -> list[FloatArray]:
        return [np.zeros(shape) for rows, fan_in in self.projection_shapes for shape in ((rows, fan_in), (rows,))]

    def _rows(self, batch: FloatArray, width: int) -> FloatArray:
        # an (N, T * width) batch, or any stack of the same values, as (N * T, width) rows
        return batch.reshape(-1, width)

    def _heads(self, rows: FloatArray, tokens: int | None = None) -> FloatArray:
        # (N * T, h * d_k) rows as the (N, h, T, d_k) view: head i's columns, per example; a decode's
        # tokens given
        T = self.tokens if tokens is None else tokens
        return rows.reshape(-1, T, self.heads, self.key_size).transpose(0, 2, 1, 3)

    def _side_by_side(self, heads: FloatArray) -> FloatArray:
        # _heads' inverse: (N, h, T, d_k) as (N * T, h * d_k) rows (a copy when h > 1)
        return heads.transpose(0, 2, 1, 3).reshape(-1, self.width)

    def _rotated(
        self, rows: FloatArray, inverse: bool = False, start: int = 0, tokens: int | None = None
    ) -> FloatArray:
        # (N * T, h * d_k) rows, each head's features j and j + d_k / 2 rotated by the row's token's
        # angles, or back; a decode's tokens from position start
        T = self.tokens if tokens is None else tokens
        x = rows.reshape(-1, T, self.heads, self.key_size)
        half = self.key_size // 2
        c, s = self._angles(start, T)
        a, b = x[..., :half], x[..., half:]
        pairs = (a * c + b * s, b * c - a * s) if inverse else (a * c - b * s, a * s + b * c)
        return np.concatenate(pairs, axis=-1).reshape(rows.shape)

    # project, attend, combine: the forward pass's blocks

    def _project(
        self, R: FloatArray, start: int = 0, tokens: int | None = None
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        # each the product, then the bias, over all heads: (N * T, h * d_k) each; a rotary layer's
        # Q and K then rotated (a decode's tokens from position start)
        Q, K, V = R @ self.Wq.T + self.bq, R @ self.Wk.T + self.bk, R @ self.Wv.T + self.bv
        if self._rotary:
            return self._rotated(Q, start=start, tokens=tokens), self._rotated(K, start=start, tokens=tokens), V
        return Q, K, V

    def _attention_segments(self, segments: Any) -> Segments | None:
        # the segment ids attention masks by: a batch's, or with document_mask False only the pads'
        # (0), every other token one segment
        if segments is None:
            return None
        ids = np.asarray(segments, dtype=np.intp)
        return ids if self.document_mask else (ids != 0).astype(np.intp)

    def _masking(self, query_segments: Segments | None, key_segments: Segments | None, offset: int = 0) -> Masking:
        return Masking(self.causal, offset, self._band, query_segments, key_segments)

    def _draw(self, rng: np.random.Generator) -> Draw:
        # a key block's dropout mask, M = u >= dropout, every entry drawn, in the generator's order (D12)
        return lambda shape: (rng.random(shape) >= self.dropout).astype(np.float64)

    def weights(self) -> FloatArray:
        """
        The last forward pass's attention weights P, (N, h, T, T), recomputed from its m and l as
        the backward pass does, P = exp(S - m) / l, over one block of every key: quadratic in T,
        for tests and inspection only (the passes never hold it). Undropped.
        """
        tile = self._masking_now.tile(self.tokens, 0, self.tokens)
        assert tile is not None and tile[:2] == (0, self.tokens), "every query attends to its own key"
        S = _scores(self._Q, self._K, self.scale, tile[2])
        return exp(S - self._m) / self._l

    def dropout_mask(self) -> FloatArray | None:
        """
        The last forward pass's dropout mask M, (N, h, T, T), drawn again from the generator's
        saved state in the passes' order (key block by key block, D12): quadratic in T, for tests
        and inspection only. None when the forward drew none.
        """
        if self._dropout_state is None:
            return None
        draw = self._draw(_generator_at(self._dropout_state))
        N, T = self._Q.shape[0], self.tokens
        blocks = [draw((N, self.heads, T, min(T, k0 + self.block) - k0)) for k0 in range(0, T, self.block)]
        return np.concatenate(blocks, axis=-1)

    def _combine(self, H: FloatArray) -> FloatArray:
        # H as (N * T, h * d_k) rows: one product over all heads' columns
        return H @ self.Wo.T + self.bo

    def forward_batch(self, X: FloatArray, *, context: ForwardContext = NO_CONTEXT) -> FloatArray:
        Q, K, V = self._project(self._rows(X, self.features))
        self._Q, self._K, self._V = self._heads(Q), self._heads(K), self._heads(V)
        segments = self._attention_segments(context.segments)
        self._masking_now = self._masking(segments, segments)
        draw = None
        self._dropout_state = None
        if self.training and self.dropout > 0.0:
            self._dropout_state = self.rng.bit_generator.state
            draw = self._draw(self.rng)
        H, self._m, self._l = tiled_forward(
            self._Q, self._K, self._V, self.scale, self._masking_now, self.block, draw, self._keep
        )
        self._H_heads = H
        self._H = self._side_by_side(H)
        self.A = self._combine(self._H).reshape(X.shape)
        return self.A

    # decoding (the generation workplan, D2, D3)

    def start_decoding(self) -> AttentionCache:
        refuse_non_causal_decoding(self.causal)
        return AttentionCache(self.width)

    def prefill(self, X: FloatArray, cache: AttentionCache, context: ForwardContext) -> FloatArray:
        # N prompts' (N * L, d) rows, L <= T, at columns 0 to L - 1: the forward pass's blocks on L
        # tokens, masked by segment when the batch has segments
        N = context.examples
        L = X.shape[0] // N
        Q, K, V = self._project(X, tokens=L)
        segments = self._attention_segments(context.segments)
        cache.start(K.reshape(N, L, self.width), V.reshape(N, L, self.width), segments)
        masking = self._masking(segments, segments)
        H, _m, _l = tiled_forward(
            self._heads(Q, L), self._heads(K, L), self._heads(V, L), self.scale, masking, self.block, None, self._keep
        )
        return self._combine(self._side_by_side(H))

    def step(self, x: FloatArray, cache: AttentionCache, context: ForwardContext) -> FloatArray:
        # one token a row, (N, d), at context.column: each query against its row's cached keys, its
        # own appended, masking keys of another segment or a pad's but its own (D4, D8)
        N = x.shape[0]
        q, k, v = self._project(x, start=context.column, tokens=1)
        segments = self._attention_segments(context.segments)
        segments = None if segments is None else segments.reshape(N, 1)
        cache.append(k.reshape(N, 1, self.width), v.reshape(N, 1, self.width), segments, self.tokens)
        c = cache.K.shape[1]
        # the query is the cache's last column: every cached key at or before it (offset c - 1)
        query_segments = None if cache.segments is None else cache.segments[:, -1:]
        masking = self._masking(query_segments, cache.segments, offset=c - 1)
        K, V = cache.K.reshape(N * c, self.width), cache.V.reshape(N * c, self.width)
        # the cache's first key is at column context.column - (c - 1): the blocks cut where the
        # full pass's do
        H, _m, _l = tiled_forward(
            self._heads(q, 1),
            self._heads(K, c),
            self._heads(V, c),
            self.scale,
            masking,
            self.block,
            None,
            self._keep,
            origin=context.column - (c - 1),
        )
        return self._combine(self._side_by_side(H))

    # the backward pass's blocks, last first

    def _combine_backward(self, delta: FloatArray) -> FloatArray:
        # dH, (N * T, h * d_k)
        return delta @ self.Wo

    def _attend_backward(self, dH: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        # per example and head, on (N, h, T, d_k) views: dQ, dK and dV, the tiles recomputed
        state = self._dropout_state
        draw = None if state is None else self._draw(_generator_at(state))
        saved = (self._Q, self._K, self._V, self._H_heads, self._m, self._l)
        return tiled_backward(dH, saved, self.scale, self._masking_now, self.block, draw, self._keep)

    def _project_backward(self, dQ: FloatArray, dK: FloatArray, dV: FloatArray) -> FloatArray:
        # three products over all heads, summed in this order
        return (dQ @ self.Wq + dK @ self.Wk) + dV @ self.Wv

    def _backward(self, downstream: FloatArray) -> None:
        self.delta_batch = downstream
        dH = self._combine_backward(self._rows(downstream, self.features))
        dQ, dK, dV = self._attend_backward(self._heads(dH))
        self._dQ, self._dK, self._dV = self._side_by_side(dQ), self._side_by_side(dK), self._side_by_side(dV)
        if self._rotary:
            # project's backward starts by rotating back: the gradients of the unrotated Q and K
            self._dQ, self._dK = self._rotated(self._dQ, inverse=True), self._rotated(self._dK, inverse=True)
        self._dX = self._project_backward(self._dQ, self._dK, self._dV).reshape(downstream.shape)

    def downstream_batch(self, *, context: ForwardContext = NO_CONTEXT) -> FloatArray:
        return self._dX

    def accumulate_gradient_batch(
        self, input_activation_batch: FloatArray, *, context: ForwardContext = NO_CONTEXT
    ) -> None:
        X = self._rows(input_activation_batch, self.features)
        delta = self._rows(self.delta_batch, self.features)
        self.grad_Wq += self._dQ.T @ X
        self.grad_bq += sum_rows(self._dQ)
        self.grad_Wk += self._dK.T @ X
        self.grad_bk += sum_rows(self._dK)
        self.grad_Wv += self._dV.T @ X
        self.grad_bv += sum_rows(self._dV)
        self.grad_Wo += delta.T @ self._H
        self.grad_bo += sum_rows(delta)
