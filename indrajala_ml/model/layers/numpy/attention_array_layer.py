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

A causal layer (the sequence task workplan, D7) sets each score S_ij with j > i to -inf before the
max shift: the row max comes from the unmasked scores (the diagonal never is), exp(-inf) is exactly
0, so a masked weight P_ij is exactly 0 and so is its dS_ij, and the backward pass is unchanged. An
unmasked layer computes what it did before the mask.

A layer with dropout (the attention-dropout workplan, D3, D4) drops its weights in training: one
(N, h, T, T) mask M per forward batch, drawn row-major from the network's generator (example,
head, query, key), every entry drawn, the causally masked ones included, M_ij = u_ij >= dropout;
P~ = P * M / keep and H = P~ V. The backward pass takes dV = P~^T dH and dP = (dH V^T) * M / keep,
then the softmax's backward with the undropped P. In inference, and at dropout 0, nothing is
drawn and the layer computes what it did before dropout.

A rotary layer (the rotary-positions workplan, D3, D4) ends project by rotating each head's Q and K,
after their biases, by their token's position t: feature j < d_k / 2 paired with j + d_k / 2,
q'_j = q_j c - q_{j+d_k/2} s and q'_{j+d_k/2} = q_j s + q_{j+d_k/2} c, with c and s from
rotary_table, read on the (N, T, h, d_k) view of the rows. Attend reads the rotated Q and K; the
backward pass rotates dQ and dK back, dq_j = dq'_j c + dq'_{j+d_k/2} s and dq_{j+d_k/2} =
dq'_{j+d_k/2} c - dq'_j s, before project's products and the gradients. V isn't rotated. An
unrotated layer computes what it did before the rotation.

A causal layer decodes (the generation workplan, D2, D3): prefill is the forward pass's blocks on
the prompt's L <= T tokens, and keeps their rotated keys and values, (L, h * d_k) each, in an
AttentionCache; step projects one token, rotates its query and key by its position (rotary_table
extended past T), appends its key and value to the cache, dropping the oldest past T, and attends
from its query to every cached key, unmasked, since none is later. So a rotary layer's window slides
exactly: scores depend on distances up to T - 1, as in training.

The softmax's exp is this module's exp, which tests replace with another implementation's (it
isn't correctly rounded), as batch norm's tests replace sigmoid.
"""

from __future__ import annotations

import math

import numpy as np

from indrajala_ml.model.layers.array.array_parameters import AttentionProjections
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.batch_norm_array_layer import sum_rows
from indrajala_ml.model.layers.numpy.token_array_layer import BatchShaped
from indrajala_ml.model.protocols.decoding_protocols import refuse_non_causal_decoding
from indrajala_ml.model.specs.rotary_table import rotary_table

exp = np.exp


def _transpose(stack: FloatArray) -> FloatArray:
    # each matrix of a stack transposed
    return stack.swapaxes(-1, -2)


class AttentionCache:
    """One decode's rotated keys K and values V, (c, h * d_k) each, the last c <= T tokens', oldest first."""

    def __init__(self, width: int) -> None:
        self.K: FloatArray = np.zeros((0, width))
        self.V: FloatArray = np.zeros((0, width))

    def append(self, K: FloatArray, V: FloatArray, window: int) -> None:
        self.K = np.concatenate((self.K, K))[-window:]
        self.V = np.concatenate((self.V, V))[-window:]


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
    ) -> None:
        self._set_up(tokens, features, heads, key_size)
        # computed once and divided by, never multiplied by its reciprocal
        self.scale = math.sqrt(self.key_size)
        self.causal = causal
        # the masked scores, j > i: True above the diagonal
        self._future = np.triu(np.ones((tokens, tokens), dtype=np.bool_), k=1)
        assert 0.0 <= dropout < 1.0, f"an attention's dropout is in [0.0, 1.0); got {dropout}"
        self.dropout = dropout
        self._keep = 1.0 - dropout
        # set by set_training_mode and set_rng, as a dense dropout layer's; the forward pass's mask,
        # None when it drew none, is what the backward pass reads
        self.training = False
        self.rng = np.random.default_rng()
        self._M: FloatArray | None = None
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

    def _attend(
        self, Q: FloatArray, K: FloatArray, V: FloatArray, masked: bool = True
    ) -> tuple[FloatArray, FloatArray]:
        # per example and head, on (N, h, T, d_k) views: the weights P (N, h, T, T) and H; a causal
        # layer masks unless the keys are all at or before the queries (a decode's step)
        S = (Q @ _transpose(K)) / self.scale
        if self.causal and masked:
            T = S.shape[-1]
            S = np.where(self._future[:T, :T], -np.inf, S)
        m = S.max(axis=-1, keepdims=True)
        e = exp(S - m)
        P = e / np.cumsum(e, axis=-1)[..., -1:]
        if self.training and self.dropout > 0.0:
            self._M = (self.rng.random(P.shape) >= self.dropout).astype(np.float64)
            self._P_dropped = P * self._M / self._keep
            return P, self._P_dropped @ V
        self._M = None
        return P, P @ V

    def _combine(self, H: FloatArray) -> FloatArray:
        # H as (N * T, h * d_k) rows: one product over all heads' columns
        return H @ self.Wo.T + self.bo

    def forward_batch(self, X: FloatArray) -> FloatArray:
        Q, K, V = self._project(self._rows(X, self.features))
        self._Q, self._K, self._V = self._heads(Q), self._heads(K), self._heads(V)
        self._P, H = self._attend(self._Q, self._K, self._V)
        self._H = self._side_by_side(H)
        self.A = self._combine(self._H).reshape(X.shape)
        return self.A

    # decoding (the generation workplan, D2, D3)

    def start_decoding(self) -> AttentionCache:
        refuse_non_causal_decoding(self.causal)
        return AttentionCache(self.width)

    def prefill(self, X: FloatArray, cache: AttentionCache) -> FloatArray:
        # the prompt's (L, d) rows, L <= T, at positions 0 to L - 1: the forward pass's blocks on L tokens
        L = X.shape[0]
        Q, K, V = self._project(X, tokens=L)
        cache.append(K, V, self.tokens)
        _P, H = self._attend(self._heads(Q, L), self._heads(K, L), self._heads(V, L))
        return self._combine(self._side_by_side(H))

    def step(self, x: FloatArray, cache: AttentionCache, position: int) -> FloatArray:
        # one token's (1, d) row at position: its query against the cached keys, its own appended
        q, k, v = self._project(x, start=position, tokens=1)
        cache.append(k, v, self.tokens)
        c = cache.K.shape[0]
        _P, H = self._attend(self._heads(q, 1), self._heads(cache.K, c), self._heads(cache.V, c), masked=False)
        return self._combine(self._side_by_side(H))

    # the backward pass's blocks, last first

    def _combine_backward(self, delta: FloatArray) -> FloatArray:
        # dH, (N * T, h * d_k)
        return delta @ self.Wo

    def _attend_backward(self, dH: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        # per example and head, on (N, h, T, d_k) views: dQ, dK and dV
        Q, K, V, P = self._Q, self._K, self._V, self._P
        dP = dH @ _transpose(V)
        if self._M is None:
            dV = _transpose(P) @ dH
        else:
            dV = _transpose(self._P_dropped) @ dH
            dP = dP * self._M / self._keep
        r = np.cumsum(dP * P, axis=-1)[..., -1:]
        dS = P * (dP - r)
        dQ = (dS @ K) / self.scale
        dK = (_transpose(dS) @ Q) / self.scale
        return dQ, dK, dV

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

    def downstream_batch(self) -> FloatArray:
        return self._dX

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
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
