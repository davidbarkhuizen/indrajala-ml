# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, Q, K, V, P, H, which strict mode takes for constants)
"""
Single-head self-attention in numpy (the layer-norm and attention workplan, D6; README, Layer norm
and attention): every expression the README's, in its grouping. The products are numpy's, on the
(N * T, d) rows for the projections and on (N, T, T) and (N, T, d) stacks between activations;
every other sum is a left fold through np.cumsum (tests/model/layers/test_summation_order.py).

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
    # each matrix of an (N, T, d) stack transposed
    return stack.transpose(0, 2, 1)


class AttentionArrayLayer(BatchShaped, AttentionProjections[FloatArray]):
    """
    Q = X Wq^T + bq, K and V likewise, P = softmax_rows((Q K^T) / sqrt(d)), out = (P V) Wo^T + bo,
    per example over its tokens tokens of features features. Its parameters, also its draw order,
    are Wq, bq, Wk, bk, Wv, bv, Wo, bo: each projection (d, d) and drawn as a dense layer's W then b,
    the weights decayed and the biases not. Hidden only, ending a token block's body.

    The backward pass runs whole in _backward, since the gradients need dQ, dK and dV and the
    layer before reads dX.
    """

    def __init__(self, tokens: int, features: int) -> None:
        self.tokens = tokens
        self.features = features
        self.size = tokens * features
        self.input_size = self.size
        self.scale = math.sqrt(features)

        d = features
        self.Wq, self.Wk, self.Wv, self.Wo = (np.zeros((d, d)) for _ in range(4))
        self.bq, self.bk, self.bv, self.bo = (np.zeros(d) for _ in range(4))
        self.reset_gradient_accum()

    def _rows(self, batch: FloatArray) -> FloatArray:
        # an (N, T * d) batch or an (N, T, d) stack as (N * T, d) rows
        return batch.reshape(-1, self.features)

    def _stack(self, rows: FloatArray) -> FloatArray:
        return rows.reshape(-1, self.tokens, self.features)

    def forward_batch(self, X: FloatArray) -> FloatArray:
        R = self._rows(X)
        # each the product, then the bias
        self._Q = self._stack(R @ self.Wq.T + self.bq)
        self._K = self._stack(R @ self.Wk.T + self.bk)
        self._V = self._stack(R @ self.Wv.T + self.bv)
        S = (self._Q @ _transpose(self._K)) / self.scale
        m = S.max(axis=2, keepdims=True)
        e = exp(S - m)
        self._P = e / np.cumsum(e, axis=2)[:, :, -1:]
        self._H = self._P @ self._V
        self.A = (self._rows(self._H) @ self.Wo.T + self.bo).reshape(X.shape)
        return self.A

    def _backward(self, downstream: FloatArray) -> None:
        self.delta_batch = downstream
        delta = self._rows(downstream)
        dH = self._stack(delta @ self.Wo)
        dP = dH @ _transpose(self._V)
        dV = _transpose(self._P) @ dH
        r = np.cumsum(dP * self._P, axis=2)[:, :, -1:]
        dS = self._P * (dP - r)
        dQ = (dS @ self._K) / self.scale
        dK = (_transpose(dS) @ self._Q) / self.scale
        self._dQ, self._dK, self._dV = self._rows(dQ), self._rows(dK), self._rows(dV)
        # three products, summed in this order
        dX = (self._dQ @ self.Wq + self._dK @ self.Wk) + self._dV @ self.Wv
        self._dX = dX.reshape(downstream.shape)

    def downstream_batch(self) -> FloatArray:
        return self._dX

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        X = self._rows(input_activation_batch)
        delta = self._rows(self.delta_batch)
        self.grad_Wq += self._dQ.T @ X
        self.grad_bq += sum_rows(self._dQ)
        self.grad_Wk += self._dK.T @ X
        self.grad_bk += sum_rows(self._dK)
        self.grad_Wv += self._dV.T @ X
        self.grad_bv += sum_rows(self._dV)
        self.grad_Wo += delta.T @ self._rows(self._H)
        self.grad_bo += sum_rows(delta)

    def reset_gradient_accum(self) -> None:
        d = self.features
        self.grad_Wq, self.grad_Wk, self.grad_Wv, self.grad_Wo = (np.zeros((d, d)) for _ in range(4))
        self.grad_bq, self.grad_bk, self.grad_bv, self.grad_bo = (np.zeros(d) for _ in range(4))
