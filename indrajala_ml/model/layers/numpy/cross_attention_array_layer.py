# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, Q, K, V, M, H, which strict mode takes for constants)
"""
Cross-attention in numpy (the cross-attention workplan, D3; Vaswani et al. 2017, §3.2.3): queries
from the decoder's tokens, keys and values from the encoder's output, its memory, which the forward
context carries (context.memory, (N, S * d_m), and context.memory_segments, (N, S)). Every block is
self-attention's (attention_array_layer.py) but project's: Q = X Wq^T + bq from the tokens' rows,
K = M Wk^T + bk and V = M Wv^T + bv from the memory's, M's rows d_m wide, which may differ from d.
The tiled attend and its backward are self-attention's, over T queries and S keys.

The mask is by segment alone (Masking.own_column False): a query reads the keys of its own segment,
never a pad's, so in a packed window a target token reads only its own pair's source (D3, D11). A
pad's query, segment 0, reads no key: its H is 0, and its row's output the bias bo. Never causal
and never rotary: the two sequences' positions are unrelated. A batch without segments, neither
side's, reads every key.

The backward pass gives dX = dQ Wq, the gradient to the tokens, and the memory's, dM = dK Wk + dV
Wv (memory_downstream_batch), which the network sums over its CrossAttention layers and runs back
through the encoder (D2). Dropout on the attention weights is self-attention's: the same draws in
the same order, again in the backward pass.
"""

from __future__ import annotations

import numpy as np

from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.attention_array_layer import (
    BLOCK,
    AttentionArrayLayer,
    AttentionCache,
    Masking,
    Segments,
    exp,
    expand,
    tiled_forward,
)
from indrajala_ml.model.layers.numpy.batch_norm_array_layer import sum_rows


class CrossAttentionArrayLayer(AttentionArrayLayer):
    """
    Per example, tokens tokens of features features (d) attending to memory_tokens memory tokens
    (S) of memory_features features (d_m), in heads heads of key_size features (d / h when None),
    sharing key_value_heads key/value heads: Q = X Wq^T + bq, K = M Wk^T + bk, V = M Wv^T + bv, each
    head's P[i] = softmax_rows((Q[i] K[j]^T) / sqrt(d_k)) over its segment's keys, H[i] = P[i] V[j],
    out = H Wo^T + bo. Its parameters, also its draw order, are self-attention's, Wq (h * d_k, d),
    Wk and Wv (g * d_k, d_m), Wo (d, h * d_k). Hidden only, ending a decoder block's body.
    """

    def __init__(
        self,
        tokens: int,
        features: int,
        memory_tokens: int,
        memory_features: int,
        heads: int = 1,
        key_size: int | None = None,
        key_value_heads: int | None = None,
        dropout: float = 0.0,
        block: int = BLOCK,
    ) -> None:
        # set before the parameters, whose shapes read them
        self.memory_tokens = memory_tokens
        self.memory_features = memory_features
        super().__init__(
            tokens,
            features,
            heads,
            key_size,
            causal=False,
            dropout=dropout,
            block=block,
            key_value_heads=key_value_heads,
        )

    @property
    def projection_shapes(self) -> tuple[tuple[int, int], ...]:
        # Wq reads the tokens, Wk and Wv the memory, Wo gives the tokens' width
        return (
            ((self.width, self.features),)
            + ((self.kv_width, self.memory_features),) * 2
            + ((self.features, self.width),)
        )

    def _masking(self, query_segments: Segments | None, key_segments: Segments | None, offset: int = 0) -> Masking:
        return Masking(False, offset, None, query_segments, key_segments, own_column=False)

    def forward_batch(self, X: FloatArray, *, context: ForwardContext = NO_CONTEXT) -> FloatArray:
        assert context.memory is not None, "a CrossAttention reads the encoder's output, the context's memory (D2)"
        R = self._rows(X, self.features)
        memory: FloatArray = np.asarray(context.memory, dtype=np.float64)
        self._memory = self._rows(memory, self.memory_features)
        Q = R @ self.Wq.T + self.bq
        K = self._memory @ self.Wk.T + self.bk
        V = self._memory @ self.Wv.T + self.bv
        self._Q = self._heads(Q)
        self._K, self._V = self._heads(K, self.memory_tokens), self._heads(V, self.memory_tokens)
        queries, keys = context.segments, context.memory_segments
        if (queries is None) != (keys is None):
            # one side without segments: its every token in segment 1, the other's own kept
            queries = np.ones(X.shape[0:1] + (self.tokens,), dtype=np.intp) if queries is None else queries
            keys = np.ones(X.shape[0:1] + (self.memory_tokens,), dtype=np.intp) if keys is None else keys
        self._masking_now = self._masking(
            None if queries is None else np.asarray(queries, dtype=np.intp),
            None if keys is None else np.asarray(keys, dtype=np.intp),
        )
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

    def weights(self) -> FloatArray:
        """
        The last forward pass's attention weights P, (N, h, T, S), recomputed from its m and l over
        one block of every key, a pad query's row 0: for tests and inspection only. Undropped.
        """
        tile = self._masking_now.tile(self.tokens, 0, self.memory_tokens)
        N = self._Q.shape[0]
        P = np.zeros((N, self.heads, self.tokens, self.memory_tokens))
        if tile is None:
            return P
        r0, r1, mask = tile
        S = (self._Q[:, :, r0:r1] @ expand(self._K, self.heads).swapaxes(-1, -2)) / self.scale
        if mask is not None:
            S = np.where(mask, S, -np.inf)
        m = np.where(self._m == -np.inf, 0.0, self._m)[:, :, r0:r1]
        l = np.where(self._l == 0.0, 1.0, self._l)[:, :, r0:r1]
        P[:, :, r0:r1] = exp(S - m) / l
        return P

    def start_decoding(self) -> AttentionCache:
        raise NotImplementedError("a CrossAttention decodes with the source encoded once (D7): stage 8")

    def _backward(self, downstream: FloatArray) -> None:
        self.delta_batch = downstream
        dH = self._combine_backward(self._rows(downstream, self.features))
        dQ, dK, dV = self._attend_backward(self._heads(dH))
        self._dQ, self._dK, self._dV = self._side_by_side(dQ), self._side_by_side(dK), self._side_by_side(dV)
        self._dX = (self._dQ @ self.Wq).reshape(downstream.shape)
        self._dM = self._dK @ self.Wk + self._dV @ self.Wv

    def memory_downstream_batch(self) -> FloatArray:
        """The gradient to the memory, (N * S, d_m) rows: dK Wk + dV Wv, summed in that order."""
        return self._dM

    def accumulate_gradient_batch(
        self, input_activation_batch: FloatArray, *, context: ForwardContext = NO_CONTEXT
    ) -> None:
        X = self._rows(input_activation_batch, self.features)
        delta = self._rows(self.delta_batch, self.features)
        self.grad_Wq += self._dQ.T @ X
        self.grad_bq += sum_rows(self._dQ)
        self.grad_Wk += self._dK.T @ self._memory
        self.grad_bk += sum_rows(self._dK)
        self.grad_Wv += self._dV.T @ self._memory
        self.grad_bv += sum_rows(self._dV)
        self.grad_Wo += delta.T @ self._H
        self.grad_bo += sum_rows(delta)
