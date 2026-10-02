# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, A, which strict mode takes for constants)
"""
A patch model's token layers in numpy (the layer-norm and attention workplan; README, Layer norm and
attention): Patches, Position, TokenMean and the token-wise dense layer. A token sequence of T
tokens of d features is flat and token-major, index t * d + j (D2), so a batch's (N, T * d)
activations are an (N * T, d) matrix without a copy.

Each of these layers, layer norm's and attention's computes a batch; its single-example methods are
the batch ones on a batch of one (BatchShaped), so learn and learn_batch of one example agree by
construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import numpy as np

from indrajala_ml.model.layers.numpy.array_layer import ArrayLayer, FloatArray
from indrajala_ml.model.layers.numpy.batch_norm_array_layer import sum_rows


class BatchShaped:
    """
    The single-example methods as the batch ones on a batch of one. A subclass computes its delta
    from the next layer's downstream in _backward; the layer is hidden.
    """

    def forward_batch(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def _backward(self, downstream: FloatArray) -> None:
        raise NotImplementedError

    def downstream_batch(self) -> FloatArray:
        raise NotImplementedError

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        raise NotImplementedError

    def forward(self, x: FloatArray) -> FloatArray:
        return self.forward_batch(x[np.newaxis, :])[0]

    def compute_output_delta(self, reference: FloatArray) -> None:
        raise NotImplementedError(f"a {type(self).__name__} is hidden, inside a network")

    def compute_output_delta_batch(self, reference_batch: FloatArray) -> None:
        raise NotImplementedError(f"a {type(self).__name__} is hidden, inside a network")

    def compute_hidden_delta(self, next_layer: Any) -> None:
        self._backward(next_layer.downstream()[np.newaxis, :])

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        self._backward(next_layer.downstream_batch())

    def downstream(self) -> FloatArray:
        return self.downstream_batch()[0]

    def accumulate_gradient(self, input_activation: FloatArray) -> None:
        self.accumulate_gradient_batch(input_activation[np.newaxis, :])


class _ParameterFree(BatchShaped):
    """A parameter-free layer whose delta is its downstream, the identity's derivative being 1."""

    delta_batch: FloatArray

    def _backward(self, downstream: FloatArray) -> None:
        self.delta_batch = downstream

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        pass


class PatchesArrayLayer(_ParameterFree):
    """
    An (H, W, C) image, channel-major (c * H * W + h * W + w), as T = H/p * W/p tokens of
    p * p * C features: token u * (W/p) + v is patch row u, column v, and its feature
    c * p * p + i * p + j is image[c, u * p + i, v * p + j] (D3). A fixed permutation; its
    downstream is the inverse permutation of its delta.
    """

    def __init__(self, height: int, width: int, channels: int, patch_size: int) -> None:
        self.size = height * width * channels
        self.input_size = self.size
        self.patch_size = patch_size
        self.channels = channels
        self.rows = height // patch_size
        self.columns = width // patch_size

    def forward_batch(self, X: FloatArray) -> FloatArray:
        n, p = X.shape[0], self.patch_size
        # (N, C, U, p, V, p) to (N, U, V, C, p, p)
        images = X.reshape(n, self.channels, self.rows, p, self.columns, p)
        return images.transpose(0, 2, 4, 1, 3, 5).reshape(n, self.size)

    def downstream_batch(self) -> FloatArray:
        n, p = self.delta_batch.shape[0], self.patch_size
        tokens = self.delta_batch.reshape(n, self.rows, self.columns, self.channels, p, p)
        return tokens.transpose(0, 3, 1, 4, 2, 5).reshape(n, self.size)


class TokenMeanArrayLayer(_ParameterFree):
    """
    The mean over the tokens, (T, d) to (d,) (D8): out_j = sum_t(x_tj) / T, a left fold over the
    tokens; dx_tj = delta_j / T for every t.
    """

    def __init__(self, tokens: int, features: int) -> None:
        self.size = features
        self.input_size = tokens * features
        self.tokens = tokens

    def forward_batch(self, X: FloatArray) -> FloatArray:
        n = X.shape[0]
        return np.cumsum(X.reshape(n, self.tokens, self.size), axis=1)[:, -1, :] / self.tokens

    def downstream_batch(self) -> FloatArray:
        n = self.delta_batch.shape[0]
        share = (self.delta_batch / self.tokens)[:, np.newaxis, :]
        return np.broadcast_to(share, (n, self.tokens, self.size)).reshape(n, self.input_size)


class PositionArrayLayer(BatchShaped):
    """
    A learned (T, d) table P added to the tokens (D7), starting at zero and never decayed:
    out_t = x_t + P_t, a new array. Its delta is its downstream, and P's gradient is that delta
    summed over the examples.
    """

    decayed: ClassVar[tuple[bool, ...]] = (False,)

    def __init__(self, tokens: int, features: int) -> None:
        self.size = tokens * features
        self.input_size = self.size
        self.P: FloatArray = np.zeros((tokens, features))
        self.grad_P: FloatArray = np.zeros((tokens, features))

    def parameters(self) -> tuple[FloatArray, ...]:
        return (self.P,)

    def gradients(self) -> tuple[FloatArray, ...]:
        return (self.grad_P,)

    def set_parameters(self, parameters: Sequence[FloatArray]) -> None:
        (self.P,) = parameters

    def forward_batch(self, X: FloatArray) -> FloatArray:
        return X + self.P.reshape(self.size)

    def _backward(self, downstream: FloatArray) -> None:
        self.delta_batch = downstream

    def downstream_batch(self) -> FloatArray:
        return self.delta_batch

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        self.grad_P += sum_rows(self.delta_batch).reshape(self.P.shape)

    def reset_gradient_accum(self) -> None:
        self.grad_P = np.zeros(self.P.shape)


class TokenDenseArrayLayer(BatchShaped, ArrayLayer):
    """
    A dense layer acting on each token, its W (size, input_size) and b shared over the tokens (D4):
    ReLUArrayLayer's or AffineArrayLayer's expressions on the (N * T, input_size) rows. Its W and b
    are drawn and stepped as a dense layer's.
    """

    def __init__(self, size: int, input_size: int, tokens: int, activation: Literal["relu", "linear"]) -> None:
        super().__init__(size, input_size)
        self.tokens = tokens
        self.activation = activation

    def forward_batch(self, X: FloatArray) -> FloatArray:
        n = X.shape[0]
        Z = X.reshape(n * self.tokens, self.input_size) @ self.W.T + self.b
        self.A = (np.maximum(0.0, Z) if self.activation == "relu" else Z).reshape(n, self.tokens * self.size)
        return self.A

    def _backward(self, downstream: FloatArray) -> None:
        self.delta_batch = downstream * (self.A > 0.0) if self.activation == "relu" else downstream

    def downstream_batch(self) -> FloatArray:
        n = self.delta_batch.shape[0]
        return (self._rows(self.delta_batch, self.size) @ self.W).reshape(n, self.tokens * self.input_size)

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        # ArrayLayer.accumulate_gradient_batch on the rows
        delta = self._rows(self.delta_batch, self.size)
        self.grad_W += delta.T @ self._rows(input_activation_batch, self.input_size)
        self.grad_b += delta.sum(axis=0)

    def _rows(self, batch: FloatArray, features: int) -> FloatArray:
        return batch.reshape(batch.shape[0] * self.tokens, features)
