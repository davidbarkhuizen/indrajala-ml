# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, Q, K, V, P, H, which strict mode takes for constants)
"""
Single-head self-attention on the Rust backend (the layer-norm and attention workplan, D6 and D9):
one call to the crate's attention.rs per pass. The crate computes the README's expressions
(Layer norm and attention) with its own products, a dense layer's for the projections and `@`
between activations, so it matches AttentionArrayLayer by bits only where no product is involved,
as the dense layers do (the parity tests explain the gap).
"""

from __future__ import annotations

from typing import Any

import indrajala_math_rust as pa

from indrajala_ml.model.array_parameters import AttentionProjections
from indrajala_ml.model.specs.hidden_layers import Hidden


class AttentionRustArrayLayer(Hidden[pa.Array], AttentionProjections[pa.Array]):
    """
    AttentionArrayLayer on the Rust backend: Q = X Wq^T + bq, K and V likewise, P =
    softmax_rows((Q K^T) / sqrt(d)), out = (P V) Wo^T + bo, per example over its tokens tokens of
    features features. Its parameters, also its draw order, are Wq, bq, Wk, bk, Wv, bv, Wo, bo;
    the weights are decayed, the biases not. Hidden only, ending a token block's body.

    The backward pass runs whole when the delta is computed (attention_downstream_batch), since the
    gradients need dQ, dK and dV and the layer before reads dX. A single example is the same ops on
    its 1D vector, with a batch of one's bits.
    """

    def __init__(self, tokens: int, features: int) -> None:
        self.tokens = tokens
        self.features = features
        self.size = tokens * features
        self.input_size = self.size

        d = features
        self.Wq, self.Wk, self.Wv, self.Wo = (pa.Array.zeros((d, d)) for _ in range(4))
        self.bq, self.bk, self.bv, self.bo = (pa.Array.zeros(d) for _ in range(4))
        self.reset_gradient_accum()

    def forward(self, x: pa.Array) -> pa.Array:
        self.a, self._Q, self._K, self._V, self._P, self._H = pa.attention_forward(x, *self.parameters())
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A, self._Q, self._K, self._V, self._P, self._H = pa.attention_forward_batch(X, *self.parameters())
        return self.A

    def _backward(self, delta: pa.Array) -> None:
        self._dX, self._dQ, self._dK, self._dV = pa.attention_downstream_batch(
            delta, self.Wq, self.Wk, self.Wv, self.Wo, self._Q, self._K, self._V, self._P
        )

    def compute_hidden_delta(self, next_layer: Any) -> None:
        self.delta = next_layer.downstream()
        self._backward(self.delta)

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        self.delta_batch = next_layer.downstream_batch()
        self._backward(self.delta_batch)

    def downstream(self) -> pa.Array:
        return self._dX

    def downstream_batch(self) -> pa.Array:
        return self._dX

    def _accumulate(self, delta: pa.Array, X: pa.Array) -> None:
        (
            self.grad_Wq,
            self.grad_bq,
            self.grad_Wk,
            self.grad_bk,
            self.grad_Wv,
            self.grad_bv,
            self.grad_Wo,
            self.grad_bo,
        ) = pa.attention_accumulate_gradient_batch(delta, X, self._H, self._dQ, self._dK, self._dV, *self.gradients())

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        self._accumulate(self.delta, input_activation)

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        self._accumulate(self.delta_batch, input_activation_batch)

    def reset_gradient_accum(self) -> None:
        d = self.features
        self.grad_Wq, self.grad_Wk, self.grad_Wv, self.grad_Wo = (pa.Array.zeros((d, d)) for _ in range(4))
        self.grad_bq, self.grad_bk, self.grad_bv, self.grad_bo = (pa.Array.zeros(d) for _ in range(4))
