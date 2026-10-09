# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, A, which strict mode takes for constants)
"""
A patch model's token layers in numpy (the layer-norm and attention workplan; README, Layer norm and
attention): Patches, Position, TokenMean and the token-wise dense layer; and a sequence model's
(the sequence task workplan, stage 3): Embedding and the token-wise softmax output layer. A token
sequence of T tokens of d features is flat and token-major, index t * d + j (D2), so a batch's
(N, T * d) activations are an (N * T, d) matrix without a copy.

Each of these layers, layer norm's and attention's computes a batch; its single-example methods are
the batch ones on a batch of one (BatchShaped), so learn and learn_batch of one example agree by
construction.

The token-wise softmax's exp is this module's exp, which tests may replace with another
implementation's, as attention's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt

from indrajala_ml.model.layers.numpy.array_layer import ArrayLayer, FloatArray
from indrajala_ml.model.layers.numpy.batch_norm_array_layer import sum_rows

exp = np.exp


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


class EmbeddingArrayLayer(BatchShaped):
    """
    T token ids, each in [0, vocabulary), as T tokens of size features: token t is row x_t of a
    learned (vocabulary, size) table E (the sequence task workplan, D5), a new array. The first
    layer, so it sends nothing back. E's gradient is a scatter-add of its delta's rows into the rows
    they read, in row order (example by example, token by token): np.add.at, unbuffered, a left fold
    per row of E. E is drawn as a weight matrix of fan-in size, and not decayed, as P isn't.
    """

    decayed: ClassVar[tuple[bool, ...]] = (False,)

    def __init__(self, tokens: int, vocabulary: int, size: int) -> None:
        self.tokens = tokens
        self.vocabulary = vocabulary
        self.features = size
        self.size = tokens * size
        self.input_size = tokens
        self.E: FloatArray = np.zeros((vocabulary, size))
        self.grad_E: FloatArray = np.zeros((vocabulary, size))

    def parameters(self) -> tuple[FloatArray, ...]:
        return (self.E,)

    def gradients(self) -> tuple[FloatArray, ...]:
        return (self.grad_E,)

    def set_parameters(self, parameters: Sequence[FloatArray]) -> None:
        (self.E,) = parameters

    def _ids(self, X: FloatArray) -> npt.NDArray[np.intp]:
        # X's token ids, refused unless each is a whole number in [0, vocabulary)
        ids = X.astype(np.intp)
        assert np.array_equal(ids, X) and ids.min() >= 0 and ids.max() < self.vocabulary, (
            f"an Embedding reads token ids, whole numbers in [0, {self.vocabulary})"
        )
        return ids

    def forward_batch(self, X: FloatArray) -> FloatArray:
        return self.E[self._ids(X)].reshape(X.shape[0], self.size)

    def _backward(self, downstream: FloatArray) -> None:
        self.delta_batch = downstream

    def downstream_batch(self) -> FloatArray:
        raise NotImplementedError("an Embedding is the first layer: nothing reads its downstream")

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        ids = self._ids(input_activation_batch).reshape(-1)
        np.add.at(self.grad_E, ids, self.delta_batch.reshape(ids.size, self.features))

    def reset_gradient_accum(self) -> None:
        self.grad_E = np.zeros(self.E.shape)


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


class TokenSoftmaxArrayLayer(TokenDenseArrayLayer):
    """
    The token-wise output layer (the sequence task workplan, D6): a softmax output layer acting on
    each token, its W (size, input_size) and b shared over the tokens. Each token's row is
    max-shifted, exp'd and divided by its sum, a left fold. The loss is the mean of the tokens'
    cross-entropies, so the output delta is (P - Y) / T, per example.
    """

    def __init__(self, size: int, input_size: int, tokens: int) -> None:
        assert size >= 2, f"a softmax layer needs at least 2 nodes to normalize over; got size={size}"
        super().__init__(size, input_size, tokens, "linear")

    def forward_batch(self, X: FloatArray) -> FloatArray:
        n = X.shape[0]
        Z = X.reshape(n * self.tokens, self.input_size) @ self.W.T + self.b
        e = exp(Z - Z.max(axis=1, keepdims=True))
        self.A = (e / np.cumsum(e, axis=1)[:, -1:]).reshape(n, self.tokens * self.size)
        return self.A

    def compute_output_delta(self, reference: FloatArray) -> None:
        self.compute_output_delta_batch(reference[np.newaxis, :])

    def compute_output_delta_batch(self, reference_batch: FloatArray) -> None:
        self.delta_batch = (self.A - reference_batch) / self.tokens
