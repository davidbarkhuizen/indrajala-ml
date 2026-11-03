# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, Q, K, V, M, H, which strict mode takes for constants)
"""
Cross-attention in pure Python (the cross-attention workplan, D3, stage 7), the counterpart of
cross_attention_array_layer.py: queries from the decoder's tokens, keys and values from the
encoder's output, its memory, which the example's forward context carries (context.memory, one row
of S * d_m floats, and context.memory_segments, one row of S ids). Every block is self-attention's
(attention_layer.py), per scalar in its grouping, but project's: Q = X Wq^T + bq from the tokens'
rows, K = M Wk^T + bk and V = M Wv^T + bv from the memory's, M's rows d_m wide. The tiled attend
and its backward are self-attention's, over T queries and S keys.

The mask is by segment alone (Masking's own_column False): a query reads the keys of its own
segment, never a pad's, so a pad's query reads none, its H 0 and its output the bias bo (the
empty-row guard of attention_layer's passes). Never causal, never rotary. A side without segments
puts its every token in segment 1, the other's kept; with neither, every key is read. Dropout is
self-attention's: numpy's (N, h, T, b) draws per key block of the memory, N h T S a batch.

The backward pass gives dX = dQ Wq, the gradient to the tokens (downstream_sum), and the memory's,
dM = dK Wk + dV Wv, S * d_m floats (memory_downstream), which the network sums over its
CrossAttention layers and runs back through the encoder (D2).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.python.attention_layer import (
    BLOCK,
    AttentionLayer,
    Masking,
    Matrix,
    _affine,  # pyright: ignore[reportPrivateUsage]
    _times_rows,  # pyright: ignore[reportPrivateUsage]
)
from indrajala_ml.model.layers.python.token_layer import WeightRow
from indrajala_ml.model.protocols.layer_protocols import InputLayer, TrainableLayer


class CrossAttentionCache:
    """
    One decode's memory keys K and values V per row, S rows of g * d_k each, computed once at
    prefill from the context's memory (the cross-attention workplan, D7), and each row's memory
    segments, or None.
    """

    def __init__(self) -> None:
        self.K: list[Matrix] = []
        self.V: list[Matrix] = []
        self.segments: list[list[int] | None] = []


class CrossAttentionLayer(AttentionLayer):
    """
    Over tokens tokens of features features (d) attending to memory_tokens memory tokens (S) of
    memory_features features (d_m), in heads heads of key_size features, sharing key_value_heads
    key/value heads: self-attention's parameters, also their draw and snapshot order, but Wk's and
    Wv's rows read d_m features. Hidden only, ending a decoder block's body.
    """

    example_fields: ClassVar[tuple[str, ...]] = (
        *AttentionLayer.example_fields,
        "_memory",
        "_memory_segments",
        "_dM",
    )

    def __init__(
        self,
        input_layer: InputLayer,
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
        super().__init__(
            input_layer, tokens, features, heads, key_size, False, dropout, block=block, key_value_heads=key_value_heads
        )
        self.memory_tokens = memory_tokens
        self.memory_features = memory_features
        self.key_tokens = memory_tokens
        self.keys, self.values = ([WeightRow(memory_features) for _ in range(self.kv_width)] for _ in range(2))
        # this example's memory, as S rows of d_m, its segments as the masks read them, and its gradient
        self._memory: Matrix = []
        self._memory_segments: list[int] | None = None
        self._dM: list[float] = []

    def _memory_rows(self, context: ForwardContext, n: int = 0) -> Matrix:
        assert context.memory is not None, "a CrossAttention reads the encoder's output, the context's memory (D2)"
        row: Sequence[float] = context.memory[n]
        d_m = self.memory_features
        assert len(row) == self.memory_tokens * d_m, (
            f"a CrossAttention's memory is {self.memory_tokens} tokens of {d_m}; got {len(row)} values"
        )
        return [list(row[s * d_m : (s + 1) * d_m]) for s in range(self.memory_tokens)]

    def _masking(self) -> Masking:
        return self._cross_masking(self.tokens, self._segments, self._memory_segments)

    def _cross_masking(self, tokens: int, queries: list[int] | None, keys: list[int] | None) -> Masking:
        if queries is None and keys is None:
            return Masking(False)
        # a side without segments: its every token in segment 1 (numpy's rule)
        return Masking(
            False,
            0,
            ([1] * tokens if queries is None else queries, [1] * self.memory_tokens if keys is None else keys),
            own_column=False,
        )

    def forward(self, *, context: ForwardContext = NO_CONTEXT) -> None:
        self._memory = self._memory_rows(context)
        self._Q = _affine(self._inputs(), self.queries)
        self._K, self._V = _affine(self._memory, self.keys), _affine(self._memory, self.values)
        self._segments = None if context.segments is None else [int(s) for s in context.segments[0]]
        keys: Any = context.memory_segments
        self._memory_segments = None if keys is None else [int(s) for s in keys[0]]
        self._take_draws()
        self._H, self._m, self._l = self._tiled((self._Q, self._K, self._V), self._masking(), draws=self._draws)
        out = self._combine(self._H)
        for node, value in zip(self.nodes, (value for row in out for value in row)):
            node.activate(value)

    # decoding (the cross-attention workplan, D7, D15): each row's memory K and V computed once, at
    # prefill, and every query, a prefill's or a step's, against them; never causal, so each query
    # is the full pass's at its token, by bits

    def start_decoding(self) -> CrossAttentionCache:  # pyright: ignore[reportIncompatibleMethodOverride]
        return CrossAttentionCache()

    def prefill(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, X: Matrix, cache: CrossAttentionCache, context: ForwardContext
    ) -> Matrix:
        keys: Any = context.memory_segments
        for n in range(context.examples):
            memory = self._memory_rows(context, n)
            cache.K.append(_affine(memory, self.keys))
            cache.V.append(_affine(memory, self.values))
            cache.segments.append(None if keys is None else [int(s) for s in keys[n]])
        return self._attend_cached(X, cache, context)

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x: Matrix, cache: CrossAttentionCache, context: ForwardContext
    ) -> Matrix:
        return self._attend_cached(x, cache, context)

    def _attend_cached(self, X: Matrix, cache: CrossAttentionCache, context: ForwardContext) -> Matrix:
        # each row's L queries against its cached keys and values
        L = len(X) // context.examples
        out: Matrix = []
        for n in range(context.examples):
            Q = _affine(X[n * L : (n + 1) * L], self.queries)
            queries = None if context.segments is None else [int(s) for s in context.segments[n]]
            masking = self._cross_masking(L, queries, cache.segments[n])
            H, _m, _l = self._tiled((Q, cache.K[n], cache.V[n]), masking)
            out += self._combine(H)
        return out

    def compute_hidden_deltas(self, next_layer: TrainableLayer, *, context: ForwardContext = NO_CONTEXT) -> None:
        for own_index, node in enumerate(self.nodes):
            node.delta = next_layer.downstream_sum(own_index)
        dH = self._combine_backward(self._deltas())
        self._dQ, self._dK, self._dV = self._attend_backward(dH)
        self._dX = [x for row in _times_rows(self._dQ, self.queries) for x in row]
        dM_k, dM_v = _times_rows(self._dK, self.keys), _times_rows(self._dV, self.values)
        self._dM = [k + v for k_row, v_row in zip(dM_k, dM_v) for k, v in zip(k_row, v_row)]

    def memory_downstream(self) -> list[float]:
        """The gradient to this example's memory, S * d_m floats: dK Wk + dV Wv, summed in that order."""
        return self._dM

    def accumulate_gradients(self, *, context: ForwardContext = NO_CONTEXT) -> None:
        # each projection's rows over this example's tokens in order, after the examples before it:
        # Wq against the tokens, Wk and Wv against the memory, Wo against H
        X = self._inputs()
        for rows, gradient, inputs in (
            (self.queries, self._dQ, X),
            (self.keys, self._dK, self._memory),
            (self.values, self._dV, self._memory),
            (self.outputs, self._deltas(), self._H),
        ):
            for g_row, x in zip(gradient, inputs):
                for row, g in zip(rows, g_row):
                    accum = row.weight_gradient_accum
                    for j, x_j in enumerate(x):
                        accum[j] += g * x_j
                    row.bias_gradient_accum += g
