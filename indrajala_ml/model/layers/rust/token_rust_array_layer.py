# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, A, which strict mode takes for constants)
"""
A patch model's token layers on the Rust backend (the layer-norm and attention workplan; README,
Layer norm and attention): Patches, Position, TokenMean and the token-wise dense layer; and a
sequence model's (the sequence task workplan, stage 5): Embedding and the token-wise softmax output
layer; and the token-wise dropout (the attention-dropout workplan, stage 5), the counterparts of
token_array_layer.py's. Patches, TokenMean and Embedding are the crate's
tokens.rs; Position is Array's + and sum_axis0; the token-wise dense and softmax layers are the
existing dense ops (fused.rs) on the (N * T, d) rows (D9), the softmax's row sum a left fold
(array_softmax), as numpy's. Each takes a single example as its 1D vector, with a batch of one's
bits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import indrajala_math_rust as pa

from indrajala_ml.model.layers.array.array_parameters import WeightAndBias
from indrajala_ml.model.layers.python.activations import Smooth, TokenActivation
from indrajala_ml.model.protocols.decoding_protocols import RowDecoding
from indrajala_ml.model.specs.hidden_layers import DeltaIsDownstream, Hidden, ParameterFree
from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext


class PatchesRustArrayLayer(Hidden[pa.Array], DeltaIsDownstream[pa.Array], ParameterFree[pa.Array]):
    """
    PatchesArrayLayer on the Rust backend: each channel-major (H, W, C) image as H/p * W/p tokens of
    p * p * C features (patches_forward), its downstream the inverse permutation (patches_downstream).
    """

    def __init__(self, height: int, width: int, channels: int, patch_size: int) -> None:
        self.size = height * width * channels
        self.input_size = self.size
        self.height = height
        self.width = width
        self.channels = channels
        self.patch_size = patch_size

    def forward(self, x: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return pa.patches_forward(x, self.height, self.width, self.channels, self.patch_size)

    def forward_batch(self, X: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return pa.patches_forward(X, self.height, self.width, self.channels, self.patch_size)

    def downstream(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return pa.patches_downstream(self.delta, self.height, self.width, self.channels, self.patch_size)

    def downstream_batch(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return pa.patches_downstream(self.delta_batch, self.height, self.width, self.channels, self.patch_size)


class TokenMeanRustArrayLayer(Hidden[pa.Array], DeltaIsDownstream[pa.Array], ParameterFree[pa.Array]):
    """TokenMeanArrayLayer on the Rust backend: the mean over the tokens, (T, d) to (d,)."""

    def __init__(self, tokens: int, features: int) -> None:
        self.size = features
        self.input_size = tokens * features
        self.tokens = tokens

    def forward(self, x: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return pa.token_mean_forward(x, self.tokens)

    def forward_batch(self, X: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return pa.token_mean_forward(X, self.tokens)

    def downstream(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return pa.token_mean_downstream(self.delta, self.tokens)

    def downstream_batch(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return pa.token_mean_downstream(self.delta_batch, self.tokens)


class TokenDropoutRustArrayLayer(
    RowDecoding[pa.Array], Hidden[pa.Array], DeltaIsDownstream[pa.Array], ParameterFree[pa.Array]
):
    """
    TokenDropoutArrayLayer on the Rust backend: in training each forward pass draws its mask from
    rng, the network's crate Generator (token_dropout_forward), numpy's default_rng, so from
    generators in the same state the masks are TokenDropoutArrayLayer's by bits; the downstream is
    token_dropout_downstream with that mask. In inference it passes its input on and draws nothing.
    """

    def __init__(self, tokens: int, features: int, drop_probability: float) -> None:
        assert 0.0 <= drop_probability < 1.0, f"drop_probability must be in [0.0, 1.0); got {drop_probability}"
        self.size = tokens * features
        self.input_size = self.size
        self._drop_probability = drop_probability
        self.training = False
        self.rng = pa.default_rng()
        self._mask: pa.Array | None = None

    @property
    def drop_probability(self) -> float:
        return self._drop_probability

    def set_rng(self, rng: pa.Generator) -> None:
        self.rng = rng

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def forward(self, x: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        if not self.training:
            self._mask = None
            return x
        a, self._mask = pa.token_dropout_forward(x, self._drop_probability, self.rng)
        return a

    def forward_batch(self, X: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return self.forward(X, context=context)

    def decode_rows(self, R: pa.Array) -> pa.Array:
        # the identity, as in inference
        return R

    def _downstream(self, delta: pa.Array) -> pa.Array:
        return delta if self._mask is None else pa.token_dropout_downstream(delta, self._mask, self._drop_probability)

    def downstream(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return self._downstream(self.delta)

    def downstream_batch(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return self._downstream(self.delta_batch)


class EmbeddingRustArrayLayer(RowDecoding[pa.Array], Hidden[pa.Array], DeltaIsDownstream[pa.Array]):
    """
    EmbeddingArrayLayer on the Rust backend: T token ids as T tokens of size features, token t row
    x_t of a learned (vocabulary, size) table E (embedding_forward), which refuses any id that isn't
    a whole number in [0, vocabulary). The first layer, so it sends nothing back. E's gradient is a
    scatter-add of its delta's rows into the rows they read, in row order (embedding_accumulate_gradient,
    np.add.at's bits). E is drawn as a weight matrix of fan-in size, and not decayed.
    """

    decayed: ClassVar[tuple[bool, ...]] = (False,)

    def __init__(self, tokens: int, vocabulary: int, size: int, pad: int | None = None) -> None:
        # the pad's id when the Embedding has segments: its row stays zero (the segments workplan, D2)
        self.pad = pad
        self.tokens = tokens
        self.vocabulary = vocabulary
        self.features = size
        self.size = tokens * size
        self.input_size = tokens
        self.E = pa.Array.zeros((vocabulary, size))
        self.reset_gradient_accum()

    def parameters(self) -> tuple[pa.Array, ...]:
        return (self.E,)

    def gradients(self) -> tuple[pa.Array, ...]:
        return (self.grad_E,)

    def set_parameters(self, parameters: Sequence[pa.Array]) -> None:
        (self.E,) = parameters

    def forward(self, x: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return pa.embedding_forward(x, self.E)

    def forward_batch(self, X: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return pa.embedding_forward(X, self.E)

    def decode_rows(self, R: pa.Array) -> pa.Array:
        # L ids, an (L,) vector, as their (L, size) rows
        return pa.embedding_forward(R, self.E).reshape((R.shape[0], self.features))

    def downstream(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        raise NotImplementedError("an Embedding is the first layer: nothing reads its downstream")

    def downstream_batch(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        raise NotImplementedError("an Embedding is the first layer: nothing reads its downstream")

    def accumulate_gradient(self, input_activation: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.grad_E = pa.embedding_accumulate_gradient(self.delta, input_activation, self.grad_E)
        self._zero_pad_gradient()

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.grad_E = pa.embedding_accumulate_gradient(self.delta_batch, input_activation_batch, self.grad_E)
        self._zero_pad_gradient()

    def _zero_pad_gradient(self) -> None:
        if self.pad is not None:
            for j in range(self.features):
                self.grad_E[self.pad, j] = 0.0  # the pad's row is never updated

    def reset_gradient_accum(self) -> None:
        self.grad_E = pa.Array.zeros((self.vocabulary, self.features))


class PositionRustArrayLayer(Hidden[pa.Array], DeltaIsDownstream[pa.Array]):
    """
    PositionArrayLayer on the Rust backend: a learned (T, d) table P added to the tokens, starting
    at zero and never decayed. Its delta is its downstream, and P's gradient that delta summed over
    the examples (sum_axis0, a left fold).
    """

    decayed: ClassVar[tuple[bool, ...]] = (False,)

    def __init__(self, tokens: int, features: int) -> None:
        self.size = tokens * features
        self.input_size = self.size
        self.tokens = tokens
        self.features = features
        self.P = pa.Array.zeros((tokens, features))
        self.reset_gradient_accum()

    def parameters(self) -> tuple[pa.Array, ...]:
        return (self.P,)

    def gradients(self) -> tuple[pa.Array, ...]:
        return (self.grad_P,)

    def set_parameters(self, parameters: Sequence[pa.Array]) -> None:
        (self.P,) = parameters

    def forward(self, x: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        if context.positions is not None:
            return x + pa.embedding_forward(context.positions, self.P).reshape(self.size)
        return x + self.P.reshape(self.size)

    def forward_batch(self, X: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        # a new array: X is not written
        if context.positions is not None:
            # each token's row by its position within its segment (the segments workplan, D5): an
            # embedding_forward gather with the positions as ids
            return X + pa.embedding_forward(context.positions, self.P)
        return X + self.P.reshape(self.size)

    def start_decoding(self) -> None:
        return None

    def prefill(self, X: pa.Array, cache: Any, context: ForwardContext) -> pa.Array:
        # N prompts' (N * L, d) rows plus each its position's row of P, or, with no segments, rows 0 to L - 1 each
        if context.positions is not None:
            positions = [int(t) for row in context.positions.tolist() for t in row]
        else:
            positions = list(range(X.shape[0] // context.examples)) * context.examples
        return X + self.P.take_rows(positions)

    def step(self, x: pa.Array, cache: Any, context: ForwardContext) -> pa.Array:
        # one token a row plus its position's row, or, with no segments, the column's
        if context.positions is not None:
            positions = [int(row[0]) for row in context.positions.tolist()]
        else:
            positions = [context.column] * x.shape[0]
        assert max(positions) < self.tokens, f"a learned Position has {self.tokens} rows; got {max(positions)}"
        return x + self.P.take_rows(positions)

    def downstream(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return self.delta

    def downstream_batch(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        return self.delta_batch

    def accumulate_gradient(self, input_activation: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        if context.positions is not None:
            delta = self.delta.reshape((1, self.size))
            self.grad_P = pa.embedding_accumulate_gradient(delta, context.positions, self.grad_P)
            return
        self.grad_P = self.grad_P + self.delta.reshape((self.tokens, self.features))

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        if context.positions is not None:
            # a scatter-add into the rows the tokens read, as the Embedding's
            self.grad_P = pa.embedding_accumulate_gradient(self.delta_batch, context.positions, self.grad_P)
            return
        self.grad_P = self.grad_P + pa.sum_axis0(self.delta_batch).reshape((self.tokens, self.features))

    def reset_gradient_accum(self) -> None:
        self.grad_P = pa.Array.zeros((self.tokens, self.features))


class TokenDenseRustArrayLayer(RowDecoding[pa.Array], Hidden[pa.Array], WeightAndBias[pa.Array]):
    """
    TokenDenseArrayLayer on the Rust backend: a dense layer acting on each token, its W (size,
    input_size) and b shared over the tokens (D4), as the existing dense ops (the crate's activation
    family, layer_activation_forward_batch, affine_forward_batch, layer_downstream_batch,
    layer_accumulate_gradient_batch) on the rows of tokens. Its delta is the next layer's
    downstream, masked by array_activation_mask for ReLU (on a) or GELU (on z, which the forward
    keeps): no token layer after it has a W to read (D5's mask path).

    Not a RustArrayLayer: the optimizer's fused single-example SGD step is a flat dense layer's.
    """

    def __init__(self, size: int, input_size: int, tokens: int, activation: TokenActivation) -> None:
        self.size = size
        self.input_size = input_size
        self.tokens = tokens
        self.activation: TokenActivation = activation
        self.W = pa.Array.zeros((size, input_size))
        self.b = pa.Array.zeros(size)
        self.z: pa.Array | None = None
        self.Z: pa.Array | None = None
        self.reset_gradient_accum()

    def _rows(self, values: pa.Array, features: int) -> pa.Array:
        # one example's (T * features) or a batch's (N, T * features) as rows of features
        examples = 1 if len(values.shape) == 1 else values.shape[0]
        return values.reshape((examples * self.tokens, features))

    def _forward_rows(self, R: pa.Array) -> tuple[pa.Array, pa.Array | None]:
        # (rows, input_size) to the (rows, size) activations, and z where the derivative reads it
        if self.activation == "linear":
            return pa.affine_forward_batch(self.W, R, self.b), None
        return pa.layer_activation_forward_batch(self.W, R, self.b, self.activation)

    def forward(self, x: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        a, z = self._forward_rows(self._rows(x, self.input_size))
        self.a = a.reshape(self.tokens * self.size)
        self.z = None if z is None else z.reshape(self.tokens * self.size)
        return self.a

    def forward_batch(self, X: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        A, Z = self._forward_rows(self._rows(X, self.input_size))
        self.A = A.reshape((X.shape[0], self.tokens * self.size))
        self.Z = None if Z is None else Z.reshape((X.shape[0], self.tokens * self.size))
        return self.A

    def decode_rows(self, R: pa.Array) -> pa.Array:
        # the new tokens' (L, input_size) rows, the forward op's on its rows
        return self._forward_rows(R)[0]

    def _masked(self, downstream: pa.Array, a: pa.Array, z: pa.Array | None) -> pa.Array:
        if self.activation == "linear":
            return downstream
        return pa.array_activation_mask(downstream, a if z is None else z, self.activation)

    def compute_hidden_delta(self, next_layer: Any, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.delta = self._masked(next_layer.downstream(context=context), self.a, self.z)

    def compute_hidden_delta_batch(self, next_layer: Any, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.delta_batch = self._masked(next_layer.downstream_batch(context=context), self.A, self.Z)

    def downstream(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        rows = pa.layer_downstream_batch(self.W, self._rows(self.delta, self.size))
        return rows.reshape(self.tokens * self.input_size)

    def downstream_batch(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        rows = pa.layer_downstream_batch(self.W, self._rows(self.delta_batch, self.size))
        return rows.reshape((self.delta_batch.shape[0], self.tokens * self.input_size))

    def _accumulate(self, delta: pa.Array, X: pa.Array) -> None:
        self.grad_W, self.grad_b = pa.layer_accumulate_gradient_batch(
            self._rows(delta, self.size), self._rows(X, self.input_size), self.grad_W, self.grad_b
        )

    def accumulate_gradient(self, input_activation: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self._accumulate(self.delta, input_activation)

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self._accumulate(self.delta_batch, input_activation_batch)

    def reset_gradient_accum(self) -> None:
        self.grad_W = pa.Array.zeros((self.size, self.input_size))
        self.grad_b = pa.Array.zeros(self.size)


class GatedDenseRustArrayLayer(RowDecoding[pa.Array], Hidden[pa.Array], WeightAndBias[pa.Array]):
    """
    GatedDenseArrayLayer on the Rust backend (the FFN activations workplan, D3): gated_forward_batch
    on the rows of tokens, act(a) * v of one (2 * size, input_size) product, keeping Z; its delta,
    Z's, from gated_delta_batch; its downstream and gradients the dense ops' (layer_downstream_batch,
    layer_accumulate_gradient_batch) on that delta's rows.
    """

    def __init__(self, size: int, input_size: int, tokens: int, activation: Smooth) -> None:
        self.size = size
        self.input_size = input_size
        self.tokens = tokens
        self.activation: Smooth = activation
        self.W = pa.Array.zeros((2 * size, input_size))
        self.b = pa.Array.zeros(2 * size)
        self.reset_gradient_accum()

    def _rows(self, values: pa.Array, features: int) -> pa.Array:
        # one example's (T * features) or a batch's (N, T * features) as rows of features
        examples = 1 if len(values.shape) == 1 else values.shape[0]
        return values.reshape((examples * self.tokens, features))

    def forward(self, x: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        out, self.z = pa.gated_forward_batch(self.W, self._rows(x, self.input_size), self.b, self.activation)
        self.a = out.reshape(self.tokens * self.size)
        return self.a

    def forward_batch(self, X: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        out, self.Z = pa.gated_forward_batch(self.W, self._rows(X, self.input_size), self.b, self.activation)
        self.A = out.reshape((X.shape[0], self.tokens * self.size))
        return self.A

    def decode_rows(self, R: pa.Array) -> pa.Array:
        return pa.gated_forward_batch(self.W, R, self.b, self.activation)[0]

    def compute_hidden_delta(self, next_layer: Any, *, context: ForwardContext = NO_CONTEXT) -> None:
        # Z's delta, (T * 2 * size), as a dense layer's delta is laid out
        downstream = self._rows(next_layer.downstream(context=context), self.size)
        self.delta = pa.gated_delta_batch(downstream, self.z, self.activation).reshape(self.tokens * 2 * self.size)

    def compute_hidden_delta_batch(self, next_layer: Any, *, context: ForwardContext = NO_CONTEXT) -> None:
        downstream = next_layer.downstream_batch(context=context)
        rows = pa.gated_delta_batch(self._rows(downstream, self.size), self.Z, self.activation)
        self.delta_batch = rows.reshape((downstream.shape[0], self.tokens * 2 * self.size))

    def downstream(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        rows = pa.layer_downstream_batch(self.W, self._rows(self.delta, 2 * self.size))
        return rows.reshape(self.tokens * self.input_size)

    def downstream_batch(self, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        rows = pa.layer_downstream_batch(self.W, self._rows(self.delta_batch, 2 * self.size))
        return rows.reshape((self.delta_batch.shape[0], self.tokens * self.input_size))

    def _accumulate(self, delta: pa.Array, X: pa.Array) -> None:
        self.grad_W, self.grad_b = pa.layer_accumulate_gradient_batch(
            self._rows(delta, 2 * self.size), self._rows(X, self.input_size), self.grad_W, self.grad_b
        )

    def accumulate_gradient(self, input_activation: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self._accumulate(self.delta, input_activation)

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self._accumulate(self.delta_batch, input_activation_batch)

    def reset_gradient_accum(self) -> None:
        self.grad_W = pa.Array.zeros((2 * self.size, self.input_size))
        self.grad_b = pa.Array.zeros(2 * self.size)


class TokenSoftmaxRustArrayLayer(TokenDenseRustArrayLayer):
    """
    TokenSoftmaxArrayLayer on the Rust backend, the token-wise output layer (the sequence task
    workplan, D6): layer_softmax_forward_batch on the rows of tokens, each row max-shifted, exp'd and
    divided by its sum, a left fold (array_softmax). The loss is the mean of the tokens'
    cross-entropies, so the output delta is (P - Y) / T, per example.
    """

    def __init__(self, size: int, input_size: int, tokens: int) -> None:
        assert size >= 2, f"a softmax layer needs at least 2 nodes to normalize over; got size={size}"
        super().__init__(size, input_size, tokens, "linear")

    def _forward_rows(self, R: pa.Array) -> tuple[pa.Array, pa.Array | None]:
        return pa.layer_softmax_forward_batch(self.W, R, self.b), None

    def compute_output_delta(self, reference: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        if context.counted is not None:
            weights = self._weights(context).reshape(self.a.shape[0])
            self.delta = (self.a - reference) * 1 / context.count * weights
            return
        self.delta = (self.a - reference) / self.tokens

    def compute_output_delta_batch(self, reference_batch: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        if context.counted is not None:
            # the mean over the batch's counted targets (the segments workplan, D6): per example
            # (P - Y) * B / C, zero at a target that doesn't count
            N = reference_batch.shape[0]
            self.delta_batch = (self.A - reference_batch) * N / context.count * self._weights(context)
            return
        self.delta_batch = (self.A - reference_batch) / self.tokens

    def _weights(self, context: ForwardContext) -> pa.Array:
        # (N, T * size): each token's counted flag over its classes, an embedding_forward gather of
        # the flags (0.0 or 1.0, as ids) from a table of a zero row and a one row
        table = pa.Array([[0.0] * self.size, [1.0] * self.size])
        return pa.embedding_forward(context.counted, table)
