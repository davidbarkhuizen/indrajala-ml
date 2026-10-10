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
from typing import Any, ClassVar, Literal

import indrajala_math_rust as pa

from indrajala_ml.model.layers.array.array_parameters import WeightAndBias
from indrajala_ml.model.specs.hidden_layers import DeltaIsDownstream, Hidden, ParameterFree


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

    def forward(self, x: pa.Array) -> pa.Array:
        return pa.patches_forward(x, self.height, self.width, self.channels, self.patch_size)

    def forward_batch(self, X: pa.Array) -> pa.Array:
        return pa.patches_forward(X, self.height, self.width, self.channels, self.patch_size)

    def downstream(self) -> pa.Array:
        return pa.patches_downstream(self.delta, self.height, self.width, self.channels, self.patch_size)

    def downstream_batch(self) -> pa.Array:
        return pa.patches_downstream(self.delta_batch, self.height, self.width, self.channels, self.patch_size)


class TokenMeanRustArrayLayer(Hidden[pa.Array], DeltaIsDownstream[pa.Array], ParameterFree[pa.Array]):
    """TokenMeanArrayLayer on the Rust backend: the mean over the tokens, (T, d) to (d,)."""

    def __init__(self, tokens: int, features: int) -> None:
        self.size = features
        self.input_size = tokens * features
        self.tokens = tokens

    def forward(self, x: pa.Array) -> pa.Array:
        return pa.token_mean_forward(x, self.tokens)

    def forward_batch(self, X: pa.Array) -> pa.Array:
        return pa.token_mean_forward(X, self.tokens)

    def downstream(self) -> pa.Array:
        return pa.token_mean_downstream(self.delta, self.tokens)

    def downstream_batch(self) -> pa.Array:
        return pa.token_mean_downstream(self.delta_batch, self.tokens)


class TokenDropoutRustArrayLayer(Hidden[pa.Array], DeltaIsDownstream[pa.Array], ParameterFree[pa.Array]):
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

    def set_rng(self, rng: pa.Generator) -> None:
        self.rng = rng

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def forward(self, x: pa.Array) -> pa.Array:
        if not self.training:
            self._mask = None
            return x
        a, self._mask = pa.token_dropout_forward(x, self._drop_probability, self.rng)
        return a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        return self.forward(X)

    def _downstream(self, delta: pa.Array) -> pa.Array:
        return delta if self._mask is None else pa.token_dropout_downstream(delta, self._mask, self._drop_probability)

    def downstream(self) -> pa.Array:
        return self._downstream(self.delta)

    def downstream_batch(self) -> pa.Array:
        return self._downstream(self.delta_batch)


class EmbeddingRustArrayLayer(Hidden[pa.Array], DeltaIsDownstream[pa.Array]):
    """
    EmbeddingArrayLayer on the Rust backend: T token ids as T tokens of size features, token t row
    x_t of a learned (vocabulary, size) table E (embedding_forward), which refuses any id that isn't
    a whole number in [0, vocabulary). The first layer, so it sends nothing back. E's gradient is a
    scatter-add of its delta's rows into the rows they read, in row order (embedding_accumulate_gradient,
    np.add.at's bits). E is drawn as a weight matrix of fan-in size, and not decayed.
    """

    decayed: ClassVar[tuple[bool, ...]] = (False,)

    def __init__(self, tokens: int, vocabulary: int, size: int) -> None:
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

    def forward(self, x: pa.Array) -> pa.Array:
        return pa.embedding_forward(x, self.E)

    def forward_batch(self, X: pa.Array) -> pa.Array:
        return pa.embedding_forward(X, self.E)

    def downstream(self) -> pa.Array:
        raise NotImplementedError("an Embedding is the first layer: nothing reads its downstream")

    def downstream_batch(self) -> pa.Array:
        raise NotImplementedError("an Embedding is the first layer: nothing reads its downstream")

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        self.grad_E = pa.embedding_accumulate_gradient(self.delta, input_activation, self.grad_E)

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        self.grad_E = pa.embedding_accumulate_gradient(self.delta_batch, input_activation_batch, self.grad_E)

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

    def forward(self, x: pa.Array) -> pa.Array:
        return x + self.P.reshape(self.size)

    def forward_batch(self, X: pa.Array) -> pa.Array:
        # a new array: X is not written
        return X + self.P.reshape(self.size)

    def downstream(self) -> pa.Array:
        return self.delta

    def downstream_batch(self) -> pa.Array:
        return self.delta_batch

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        self.grad_P = self.grad_P + self.delta.reshape((self.tokens, self.features))

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        self.grad_P = self.grad_P + pa.sum_axis0(self.delta_batch).reshape((self.tokens, self.features))

    def reset_gradient_accum(self) -> None:
        self.grad_P = pa.Array.zeros((self.tokens, self.features))


class TokenDenseRustArrayLayer(Hidden[pa.Array], WeightAndBias[pa.Array]):
    """
    TokenDenseArrayLayer on the Rust backend: a dense layer acting on each token, its W (size,
    input_size) and b shared over the tokens (D4), as the existing dense ops (layer_relu_forward_batch,
    affine_forward_batch, layer_downstream_batch, layer_accumulate_gradient_batch) on the rows of
    tokens. Its delta is the next layer's downstream, masked by array_relu_mask for ReLU: no token
    layer after it has a W to read (D5's mask path).

    Not a RustArrayLayer: the optimizer's fused single-example SGD step is a flat dense layer's.
    """

    def __init__(self, size: int, input_size: int, tokens: int, activation: Literal["relu", "linear"]) -> None:
        self.size = size
        self.input_size = input_size
        self.tokens = tokens
        self.activation = activation
        self._forward_op = pa.layer_relu_forward_batch if activation == "relu" else pa.affine_forward_batch
        self.W = pa.Array.zeros((size, input_size))
        self.b = pa.Array.zeros(size)
        self.reset_gradient_accum()

    def _rows(self, values: pa.Array, features: int) -> pa.Array:
        # one example's (T * features) or a batch's (N, T * features) as rows of features
        examples = 1 if len(values.shape) == 1 else values.shape[0]
        return values.reshape((examples * self.tokens, features))

    def forward(self, x: pa.Array) -> pa.Array:
        self.a = self._forward_op(self.W, self._rows(x, self.input_size), self.b).reshape(self.tokens * self.size)
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        rows = self._forward_op(self.W, self._rows(X, self.input_size), self.b)
        self.A = rows.reshape((X.shape[0], self.tokens * self.size))
        return self.A

    def compute_hidden_delta(self, next_layer: Any) -> None:
        downstream = next_layer.downstream()
        self.delta = pa.array_relu_mask(downstream, self.a) if self.activation == "relu" else downstream

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        downstream = next_layer.downstream_batch()
        self.delta_batch = pa.array_relu_mask(downstream, self.A) if self.activation == "relu" else downstream

    def downstream(self) -> pa.Array:
        rows = pa.layer_downstream_batch(self.W, self._rows(self.delta, self.size))
        return rows.reshape(self.tokens * self.input_size)

    def downstream_batch(self) -> pa.Array:
        rows = pa.layer_downstream_batch(self.W, self._rows(self.delta_batch, self.size))
        return rows.reshape((self.delta_batch.shape[0], self.tokens * self.input_size))

    def _accumulate(self, delta: pa.Array, X: pa.Array) -> None:
        self.grad_W, self.grad_b = pa.layer_accumulate_gradient_batch(
            self._rows(delta, self.size), self._rows(X, self.input_size), self.grad_W, self.grad_b
        )

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        self._accumulate(self.delta, input_activation)

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        self._accumulate(self.delta_batch, input_activation_batch)

    def reset_gradient_accum(self) -> None:
        self.grad_W = pa.Array.zeros((self.size, self.input_size))
        self.grad_b = pa.Array.zeros(self.size)


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
        self._forward_op = pa.layer_softmax_forward_batch

    def compute_output_delta(self, reference: pa.Array) -> None:
        self.delta = (self.a - reference) / self.tokens

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        self.delta_batch = (self.A - reference_batch) / self.tokens
