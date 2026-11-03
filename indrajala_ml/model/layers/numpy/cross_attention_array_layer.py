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

from typing import Any

import numpy as np

from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.attention_array_layer import (
    BLOCK,
    AttentionArrayLayer,
    Masking,
    Segments,
    exp,
    expand,
    tiled_forward,
)
from indrajala_ml.model.layers.numpy.batch_norm_array_layer import sum_rows


class CrossAttentionCache:
    """
    One decode's memory keys K and values V, (N, g, S, d_k) each, computed once at prefill from the
    context's memory (the cross-attention workplan, D7), and the memory's segments, (N, S), or None.
    """

    def __init__(self) -> None:
        self.K: FloatArray | None = None
        self.V: FloatArray | None = None
        self.segments: Segments | None = None


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

    def _memory_keys(self, context: ForwardContext) -> tuple[FloatArray, FloatArray, FloatArray]:
        # the context's memory as (N * S, d_m) rows, and its K and V heads, (N, g, S, d_k)
        assert context.memory is not None, "a CrossAttention reads the encoder's output, the context's memory (D2)"
        memory = self._rows(np.asarray(context.memory, dtype=np.float64), self.memory_features)
        K = memory @ self.Wk.T + self.bk
        V = memory @ self.Wv.T + self.bv
        return memory, self._heads(K, self.memory_tokens), self._heads(V, self.memory_tokens)

    def _cross_masking(self, examples: int, tokens: int, queries: Any, keys: Any) -> Masking:
        # by segment alone; one side without segments has its every token in segment 1, the other's kept
        if (queries is None) != (keys is None):
            queries = np.ones((examples, tokens), dtype=np.intp) if queries is None else queries
            keys = np.ones((examples, self.memory_tokens), dtype=np.intp) if keys is None else keys
        return self._masking(
            None if queries is None else np.asarray(queries, dtype=np.intp),
            None if keys is None else np.asarray(keys, dtype=np.intp),
        )

    def forward_batch(self, X: FloatArray, *, context: ForwardContext = NO_CONTEXT) -> FloatArray:
        R = self._rows(X, self.features)
        self._memory, self._K, self._V = self._memory_keys(context)
        Q = R @ self.Wq.T + self.bq
        self._Q = self._heads(Q)
        self._masking_now = self._cross_masking(X.shape[0], self.tokens, context.segments, context.memory_segments)
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

    # decoding (the cross-attention workplan, D7, D15): the memory's K and V computed once, at
    # prefill, and every query, a prefill's or a step's, against them; never causal, so each query
    # is the full pass's at its token

    def start_decoding(self) -> CrossAttentionCache:  # pyright: ignore[reportIncompatibleMethodOverride]
        return CrossAttentionCache()

    def _attend_cached(self, X: FloatArray, cache: CrossAttentionCache, context: ForwardContext) -> FloatArray:
        # the queries' (N * L, d) rows against the cached keys and values
        assert cache.K is not None and cache.V is not None, "a decode starts with prefill"
        N = context.examples
        L = X.shape[0] // N
        Q = self._heads(X @ self.Wq.T + self.bq, L)
        queries = None if context.segments is None else np.asarray(context.segments, dtype=np.intp).reshape(N, L)
        masking = self._cross_masking(N, L, queries, cache.segments)
        H, _m, _l = tiled_forward(Q, cache.K, cache.V, self.scale, masking, self.block, None, self._keep)
        return self._combine(self._side_by_side(H))

    def prefill(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, X: FloatArray, cache: CrossAttentionCache, context: ForwardContext
    ) -> FloatArray:
        _memory, cache.K, cache.V = self._memory_keys(context)
        keys = context.memory_segments
        cache.segments = None if keys is None else np.asarray(keys, dtype=np.intp)
        return self._attend_cached(X, cache, context)

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x: FloatArray, cache: CrossAttentionCache, context: ForwardContext
    ) -> FloatArray:
        return self._attend_cached(x, cache, context)

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
