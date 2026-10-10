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

exp = np.exp


def _transpose(stack: FloatArray) -> FloatArray:
    # each matrix of a stack transposed
    return stack.swapaxes(-1, -2)


class AttentionArrayLayer(BatchShaped, AttentionProjections[FloatArray]):
    """
    Per example over its tokens tokens of features features (d), in heads heads (h) of key_size
    features (d_k, d / h when None): Q = X Wq^T + bq, K and V likewise, each head's
    P[i] = softmax_rows((Q[i] K[i]^T) / sqrt(d_k)) and H[i] = P[i] V[i], out = H Wo^T + bo with H
    the heads side by side; a causal one masks S_ij for j > i, and one with dropout drops P's
    entries in training. Its parameters, also its draw order,
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

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def _zeros(self) -> list[FloatArray]:
        return [np.zeros(shape) for rows, fan_in in self.projection_shapes for shape in ((rows, fan_in), (rows,))]

    def _rows(self, batch: FloatArray, width: int) -> FloatArray:
        # an (N, T * width) batch, or any stack of the same values, as (N * T, width) rows
        return batch.reshape(-1, width)

    def _heads(self, rows: FloatArray) -> FloatArray:
        # (N * T, h * d_k) rows as the (N, h, T, d_k) view: head i's columns, per example
        return rows.reshape(-1, self.tokens, self.heads, self.key_size).transpose(0, 2, 1, 3)

    def _side_by_side(self, heads: FloatArray) -> FloatArray:
        # _heads' inverse: (N, h, T, d_k) as (N * T, h * d_k) rows (a copy when h > 1)
        return heads.transpose(0, 2, 1, 3).reshape(-1, self.width)

    # project, attend, combine: the forward pass's blocks

    def _project(self, R: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        # each the product, then the bias, over all heads: (N * T, h * d_k) each
        return R @ self.Wq.T + self.bq, R @ self.Wk.T + self.bk, R @ self.Wv.T + self.bv

    def _attend(self, Q: FloatArray, K: FloatArray, V: FloatArray) -> tuple[FloatArray, FloatArray]:
        # per example and head, on (N, h, T, d_k) views: the weights P (N, h, T, T) and H
        S = (Q @ _transpose(K)) / self.scale
        if self.causal:
            S = np.where(self._future, -np.inf, S)
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
