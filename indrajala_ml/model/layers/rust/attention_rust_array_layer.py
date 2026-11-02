# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, Q, K, V, P, H, which strict mode takes for constants)
"""
Multi-head self-attention on the Rust backend (the multi-head attention workplan, D4): one call to
the crate's attention.rs per pass, each composed there of the blocks project, attend and combine.
The crate computes the README's expressions (Layer norm and attention) with its own products, a
dense layer's for the projections and `@` between activations, so it matches AttentionArrayLayer
by bits only where no product is involved, as the dense layers do (the parity tests explain the gap).

A causal layer decodes (the generation workplan, D2, D3, D8; the segments workplan, D8): prefill is
attention_forward_batch on a batch of prompts' L <= T tokens, with the table's first L rows and the
batch's segments, keeping each row's last segment's rotated keys and values in an
AttentionRustCache; step is the crate's attention_step per row, one token on its row's cache (emptied
first when the token starts a segment, so it holds the keys a mask would leave, D8 (b)) with its
column's row of the table (extended past T), its key and value appended and the oldest dropped past
T, and its first cached key's column (origin), so the crate cuts the cache's key blocks where the
full pass's are (the context-length workplan, D3). A batched crate step reading the keys' segments
would take numpy's form (next-steps.md). Each of the crate's sums is a sequential fold, so a
decode's bits are the full pass's within the window.
"""

from __future__ import annotations

from typing import Any

import indrajala_math_rust as pa

from indrajala_ml.model.layers.array.array_parameters import AttentionProjections
from indrajala_ml.model.protocols.decoding_protocols import refuse_non_causal_decoding
from indrajala_ml.model.specs.hidden_layers import Hidden
from indrajala_ml.model.specs.rotary_table import rotary_table
from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext


class AttentionRustCache:
    """
    One decode's rotated keys K and values V per row of its batch, (c, h * d_k) each: each row's
    current segment's, its last c <= T tokens', oldest first (the segments workplan, D8 (b)), and
    that segment's id (1 for every row of a batch without segments).
    """

    def __init__(self, width: int) -> None:
        self.width = width
        self.K: list[pa.Array] = []
        self.V: list[pa.Array] = []
        self.segments: list[int] = []

    def start_segment(self, n: int, segment: int) -> None:
        # empties row n's cache when its next token starts a segment (or is a pad's)
        if segment != self.segments[n] or segment == 0:
            self.K[n], self.V[n] = pa.Array.zeros((0, self.width)), pa.Array.zeros((0, self.width))
            self.segments[n] = segment


class AttentionRustArrayLayer(Hidden[pa.Array], AttentionProjections[pa.Array]):
    """
    AttentionArrayLayer on the Rust backend: per example over its tokens tokens of features
    features (d), in heads heads (h) of key_size features (d_k, d / h when None): Q = X Wq^T + bq,
    K and V likewise, each head's P[i] = softmax_rows((Q[i] K[i]^T) / sqrt(d_k)) and
    H[i] = P[i] V[i], out = H Wo^T + bo with H the heads side by side. Its parameters, also its
    draw order, are Wq, bq, Wk, bk, Wv, bv, Wo, bo: Wq, Wk, Wv (h * d_k, d) and Wo (d, h * d_k),
    heads as row (Wo: column) blocks; the weights are decayed, the biases not. Attend is tiled, as
    AttentionArrayLayer's (the context-length workplan, D3): the crate keeps each query's max m and
    sum l, never P, and its backward op recomputes each key block's weights, masked as the forward
    was (causal, the sequence task workplan, D7; by segment, the segments workplan, D4), so the
    layer keeps the forward's segments for it. A layer with dropout (the attention-dropout workplan,
    D3-D5; the context-length workplan, D12) drops its weights in training: the forward ops draw the
    masks from rng, the network's crate Generator, key block by key block in numpy's (N, h, T, b)
    order, so from generators in the same state they are AttentionArrayLayer's by bits, and return
    the generator's copy before the draws, from which the backward op draws them again. In inference
    and at dropout 0 nothing is drawn. A rotary layer (the rotary-positions workplan, D4, D5) holds
    rotary_table's cos and sin as crate arrays, built once, and passes them to the forward ops, which
    rotate Q and K at the end of project, and to the backward op, which rotates dQ and dK back:
    elementwise, so numpy's bits on the same Q and K. Hidden only, ending a token block's body. The
    caches are packed as the crate's: Q, K (rotated), V, H (N * T, h * d_k), head i in columns
    i * d_k.., and m and l (N * T, h).

    The backward pass runs whole when the delta is computed (attention_downstream_batch), since the
    gradients need dQ, dK and dV and the layer before reads dX. A single example is the same ops on
    its 1D vector, with a batch of one's bits.
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
        document_mask: bool = True,
        key_value_heads: int | None = None,
    ) -> None:
        assert key_value_heads is None, (
            "grouped-query attention reaches the Rust layers at the grouped-query attention workplan's stage 4; "
            f"got key_value_heads={key_value_heads}"
        )
        assert 0.0 <= dropout < 1.0, f"an attention's dropout is in [0.0, 1.0); got {dropout}"
        self.causal = causal
        # False: attention across the paragraphs, pads still masked (the context-length workplan, D8)
        self.document_mask = document_mask
        self.dropout = dropout
        self._set_up(tokens, features, heads, key_size)
        # set by set_training_mode and set_rng, as a dense dropout layer's; the forward pass's copy of
        # the generator before its draws, None when it drew none, is what the backward pass draws
        # the same masks from (the context-length workplan, D12)
        self.training = False
        self.rng = pa.default_rng()
        self._dropout_rng: pa.Generator | None = None
        self._segments: Any = None
        assert positions in ("none", "rotary"), f'an attention\'s positions are "none" or "rotary"; got {positions!r}'
        self.positions = positions
        self.rotary_base = rotary_base
        # the ops' cos and sin keywords: none for an unrotated layer
        self._table: dict[str, pa.Array] = {}
        # rotary_table's rows as lists, which a decode extends past T
        self._angles: tuple[list[list[float]], list[list[float]]] = ([], [])
        if positions == "rotary":
            self._angles = rotary_table(tokens, self.key_size, rotary_base)
            self._table = {"cos": pa.Array(self._angles[0]), "sin": pa.Array(self._angles[1])}

    @property
    def rotary_table(self) -> tuple[pa.Array, pa.Array] | tuple[None, None]:
        """The ops' cos and sin, or None for both when the layer isn't rotary."""
        return (self._table["cos"], self._table["sin"]) if self._table else (None, None)

    def set_rng(self, rng: pa.Generator) -> None:
        self.rng = rng

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def _zeros(self) -> list[pa.Array]:
        return [
            array
            for rows, fan_in in self.projection_shapes
            for array in (pa.Array.zeros((rows, fan_in)), pa.Array.zeros(rows))
        ]

    def _attention_segments(self, segments: Any) -> Any:
        # the segment ids attention masks by: a batch's, or with document_mask False only the pads'
        # (0), every other token one segment
        if segments is None or self.document_mask:
            return segments
        return pa.Array([[0.0 if s == 0.0 else 1.0 for s in row] for row in segments.tolist()])

    def forward(self, x: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        self._segments = self._attention_segments(context.segments)
        self.a, self._Q, self._K, self._V, self._H, self._m, self._l, self._dropout_rng = pa.attention_forward(
            x,
            *self.parameters(),
            heads=self.heads,
            causal=self.causal,
            dropout=self.dropout,
            training=self.training,
            rng=self.rng,
            segments=self._segments,
            **self._table,
        )
        return self.a

    def forward_batch(self, X: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        self._segments = self._attention_segments(context.segments)
        self.A, self._Q, self._K, self._V, self._H, self._m, self._l, self._dropout_rng = pa.attention_forward_batch(
            X,
            *self.parameters(),
            heads=self.heads,
            causal=self.causal,
            dropout=self.dropout,
            training=self.training,
            rng=self.rng,
            segments=self._segments,
            **self._table,
        )
        return self.A

    # decoding (the generation workplan, D2, D3)

    def _rows_table(self, start: int, tokens: int) -> dict[str, pa.Array]:
        # the ops' cos and sin for positions start to start + tokens - 1, the lists extended (doubled)
        # past their rows, which don't change; none for an unrotated layer
        if not self._table:
            return {}
        if start + tokens > len(self._angles[0]):
            positions = max(start + tokens, 2 * len(self._angles[0]))
            self._angles = rotary_table(positions, self.key_size, self.rotary_base)
        cos, sin = self._angles
        return {"cos": pa.Array(cos[start : start + tokens]), "sin": pa.Array(sin[start : start + tokens])}

    def start_decoding(self) -> AttentionRustCache:
        refuse_non_causal_decoding(self.causal)
        return AttentionRustCache(self.width)

    def prefill(self, X: pa.Array, cache: AttentionRustCache, context: ForwardContext) -> pa.Array:
        # N prompts' (N * L, d) rows, L <= T, at columns 0 to L - 1: the batch forward op on L
        # tokens, masked by segment when the batch has segments; each row keeps its last segment's
        # keys and values
        N = context.examples
        L = X.shape[0] // N
        A, _Q, K, V, _H, _m, _l, _rng = pa.attention_forward_batch(
            X.reshape((N, L * self.features)),
            *self.parameters(),
            heads=self.heads,
            causal=True,
            segments=self._attention_segments(context.segments),
            **self._rows_table(0, L),
        )
        attended = self._attention_segments(context.segments)
        rows = None if attended is None else [[int(s) for s in row] for row in attended.tolist()]
        for n in range(N):
            last = 1 if rows is None else rows[n][-1]
            keep = [n * L + t for t in range(L) if rows is None or rows[n][t] == last][-self.tokens :]
            cache.K.append(K.take_rows(keep))
            cache.V.append(V.take_rows(keep))
            cache.segments.append(last)
        return A.reshape((N * L, self.features))

    def step(self, x: pa.Array, cache: AttentionRustCache, context: ForwardContext) -> pa.Array:
        # one token a row at context.column: the crate's step per row against its row's cache, which
        # holds its segment's keys only, so unmasked (D8 (b))
        N = x.shape[0]
        attended = self._attention_segments(context.segments)
        segments = None if attended is None else [int(row[0]) for row in attended.tolist()]
        table = self._rows_table(context.column, 1)
        out: list[pa.Array] = []
        for n in range(N):
            cache.start_segment(n, 1 if segments is None else segments[n])
            # the first cached key's column once this token's is appended: the crate cuts the cache's
            # key blocks where the full pass's are (the context-length workplan, D3)
            origin = context.column + 1 - min(self.tokens, cache.K[n].shape[0] + 1)
            a, cache.K[n], cache.V[n] = pa.attention_step(
                x.row(n),
                *self.parameters(),
                k_cache=cache.K[n],
                v_cache=cache.V[n],
                heads=self.heads,
                window=self.tokens,
                origin=origin,
                **table,
            )
            out.append(a.reshape((1, self.features)))
        return pa.concatenate_columns(out).reshape((N, self.features))

    def _backward(self, delta: pa.Array) -> None:
        self._dX, self._dQ, self._dK, self._dV = pa.attention_downstream_batch(
            delta,
            self.Wq,
            self.Wk,
            self.Wv,
            self.Wo,
            self._Q,
            self._K,
            self._V,
            self._H,
            self._m,
            self._l,
            heads=self.heads,
            causal=self.causal,
            segments=self._segments,
            dropout=self.dropout,
            dropout_rng=self._dropout_rng,
            cos=self._table.get("cos"),
            sin=self._table.get("sin"),
        )

    def compute_hidden_delta(self, next_layer: Any, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.delta = next_layer.downstream(context=context)
        self._backward(self.delta)

    def compute_hidden_delta_batch(self, next_layer: Any, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.delta_batch = next_layer.downstream_batch(context=context)
        self._backward(self.delta_batch)

    def downstream(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return self._dX

    def downstream_batch(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return self._dX

    def _accumulate(self, delta: pa.Array, X: pa.Array) -> None:
        self._set_gradients(
            pa.attention_accumulate_gradient_batch(delta, X, self._H, self._dQ, self._dK, self._dV, *self.gradients())
        )

    def accumulate_gradient(self, input_activation: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self._accumulate(self.delta, input_activation)

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self._accumulate(self.delta_batch, input_activation_batch)
