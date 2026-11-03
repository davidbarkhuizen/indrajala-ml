# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, Q, K, V, M, H, which strict mode takes for constants)
"""
Cross-attention on the Rust backend (the cross-attention workplan, D3, stage 6): numpy's
CrossAttentionArrayLayer through the crate's cross_attention_* ops, one call per pass. The queries
come from the decoder's tokens, and the keys and values from the encoder's output, its memory, which
the forward context carries (context.memory, (N, S * d_m), and context.memory_segments, (N, S)).
The crate computes numpy's expressions with its own products, a dense layer's for the projections
and `@` between activations, so it matches numpy by bits only where no product is involved, as
AttentionRustArrayLayer does.

The mask is by segment alone, with no own-column exception: a query reads the keys of its own
segment and never a pad's, so a pad's query reads none, and its output is the bias bo. It is never
causal and never rotary. A batch with segments on one side only puts every token of the other in
segment 1, as numpy's does, and a batch with neither reads every key. Dropout is self-attention's:
the forward op draws from rng, the network's crate Generator, and returns its copy from before the
draws, from which the backward op draws the same masks again.

The backward pass gives dX = dQ Wq, the gradient to the tokens, and the memory's, dM = dK Wk + dV
Wv, (N * S, d_m) rows (memory_downstream_batch). The network sums dM over its CrossAttention layers
and runs it back through the encoder (D2). A single example is a batch of one.
"""

from __future__ import annotations

from typing import Any

import indrajala_math_rust as pa

from indrajala_ml.model.layers.array.array_parameters import AttentionProjections
from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.specs.hidden_layers import Hidden


class CrossAttentionRustArrayLayer(Hidden[pa.Array], AttentionProjections[pa.Array]):
    """
    Per example, tokens tokens of features features (d) attending to memory_tokens memory tokens
    (S) of memory_features features (d_m), in heads heads of key_size features (d / h when None),
    sharing key_value_heads key/value heads: Q = X Wq^T + bq, K = M Wk^T + bk, V = M Wv^T + bv, each
    head's P[i] = softmax_rows((Q[i] K[j]^T) / sqrt(d_k)) over its segment's keys, H[i] = P[i] V[j],
    out = H Wo^T + bo. Its parameters, and so their draw order, are self-attention's: Wq (h * d_k, d),
    Wk and Wv (g * d_k, d_m), Wo (d, h * d_k). Hidden only, ending a decoder block's body. The caches
    are the crate's: Q, H and m, l over the (N * T) query rows, K and V over the (N * S) memory rows.
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
    ) -> None:
        assert 0.0 <= dropout < 1.0, f"a cross-attention's dropout is in [0.0, 1.0); got {dropout}"
        # set before the parameters, whose shapes read them
        self.memory_tokens = memory_tokens
        self.memory_features = memory_features
        self.causal = False
        self.dropout = dropout
        self._set_up(tokens, features, heads, key_size, key_value_heads)
        self.training = False
        self.rng = pa.default_rng()
        self._dropout_rng: pa.Generator | None = None
        self._segments: tuple[pa.Array | None, pa.Array | None] = (None, None)

    @property
    def projection_shapes(self) -> tuple[tuple[int, int], ...]:
        # Wq reads the tokens, Wk and Wv the memory, Wo gives the tokens' width
        return (
            ((self.width, self.features),)
            + ((self.kv_width, self.memory_features),) * 2
            + ((self.features, self.width),)
        )

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

    def _segment_ids(self, examples: int, context: ForwardContext) -> tuple[pa.Array | None, pa.Array | None]:
        # the ops' segments and memory_segments: both or neither; one side without segments has its
        # every token in segment 1, the other's own kept (numpy's rule)
        queries, keys = context.segments, context.memory_segments
        if queries is None and keys is None:
            return None, None
        if queries is None:
            queries = pa.Array([[1.0] * self.tokens for _ in range(examples)])
        if keys is None:
            keys = pa.Array([[1.0] * self.memory_tokens for _ in range(examples)])
        return queries, keys

    def forward_batch(self, X: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        assert context.memory is not None, "a CrossAttention reads the encoder's output, the context's memory (D2)"
        self._memory: pa.Array = context.memory
        self._segments = self._segment_ids(X.shape[0], context)
        self.A, self._Q, self._K, self._V, self._H, self._m, self._l, self._dropout_rng = pa.cross_attention_forward_batch(
            X,
            self._memory,
            *self.parameters(),
            heads=self.heads,
            key_value_heads=self.key_value_heads,
            dropout=self.dropout,
            training=self.training,
            rng=self.rng,
            segments=self._segments[0],
            memory_segments=self._segments[1],
        )
        return self.A

    def forward(self, x: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        # one example: a batch of one
        return self.forward_batch(x.reshape((1, self.size)), context=context).reshape(self.size)

    def _backward(self, delta: pa.Array) -> None:
        self._dX, self._dM, self._dQ, self._dK, self._dV = pa.cross_attention_downstream_batch(
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
            key_value_heads=self.key_value_heads,
            dropout=self.dropout,
            dropout_rng=self._dropout_rng,
            segments=self._segments[0],
            memory_segments=self._segments[1],
        )

    def compute_hidden_delta(self, next_layer: Any, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.delta = next_layer.downstream(context=context)
        self._backward(self.delta.reshape((1, self.size)))

    def compute_hidden_delta_batch(self, next_layer: Any, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.delta_batch = next_layer.downstream_batch(context=context)
        self._backward(self.delta_batch)

    def downstream(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return self._dX.reshape(self.size)

    def downstream_batch(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return self._dX

    def memory_downstream_batch(self) -> pa.Array:
        """The gradient to the memory, (N * S, d_m) rows: dK Wk + dV Wv, summed in that order."""
        return self._dM

    def _accumulate(self, delta: pa.Array, X: pa.Array) -> None:
        self._set_gradients(
            pa.cross_attention_accumulate_gradient_batch(
                delta, X, self._memory, self._H, self._dQ, self._dK, self._dV, *self.gradients()
            )
        )

    def accumulate_gradient(self, input_activation: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self._accumulate(self.delta.reshape((1, self.size)), input_activation.reshape((1, self.size)))

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self._accumulate(self.delta_batch, input_activation_batch)
