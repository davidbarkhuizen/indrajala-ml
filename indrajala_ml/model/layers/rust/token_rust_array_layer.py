# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, A, which strict mode takes for constants)
"""
A patch model's token layers on the Rust backend (the layer-norm and attention workplan; README,
Layer norm and attention): Patches, Position, TokenMean and the token-wise dense layer, the
counterparts of token_array_layer.py's. Patches and TokenMean are the crate's tokens.rs; Position
is Array's + and sum_axis0; the token-wise dense layer is the existing dense ops (fused.rs) on the
(N * T, d) rows (D9). Each takes a single example as its 1D vector, with a batch of one's bits.
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
