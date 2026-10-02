# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, R, which strict mode takes for constants)
"""
Layer norm in numpy (the layer-norm and attention workplan, D5; README, Layer norm and attention):
each token's features, or a flat layer's as one token, normalized by their own mean and biased
variance, then gamma * xhat + beta. No batch statistics, so the same in training and inference, and
a batch of one trains as any other. Every expression is the README's, in its grouping; every sum is
a left fold through np.cumsum (tests/model/layers/test_summation_order.py): over a row's features for the
statistics and the backward pass's means, over the rows for the gradients.
"""

from __future__ import annotations

import numpy as np

from indrajala_ml.model.layers.array.array_parameters import GammaAndBeta
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.batch_norm_array_layer import sum_rows
from indrajala_ml.model.layers.numpy.token_array_layer import BatchShaped


def sum_features(R: FloatArray) -> FloatArray:
    """Each row's sum over its features, a left fold in feature order, as an (rows, 1) column."""
    return np.cumsum(R, axis=1)[:, -1:]


class LayerNormArrayLayer(BatchShaped, GammaAndBeta[FloatArray]):
    """
    y = gamma * xhat + beta over each of tokens tokens of features features, gamma and beta (features,)
    shared over the tokens, starting at 1 and 0 and never decayed. A flat layer is one token of
    its whole size, a conv front end's image included. The statistics are computed on the
    (N * tokens, features) rows, which a token-major batch is without a copy. Hidden only.
    """

    def __init__(self, tokens: int, features: int, epsilon: float) -> None:
        self.tokens = tokens
        self.features = features
        self.size = tokens * features
        self.input_size = self.size
        self.epsilon = epsilon

        # nothing drawn
        self.gamma: FloatArray = np.ones(features)
        self.beta: FloatArray = np.zeros(features)
        self.grad_gamma: FloatArray = np.zeros(features)
        self.grad_beta: FloatArray = np.zeros(features)

    def _rows(self, batch: FloatArray) -> FloatArray:
        return batch.reshape(batch.shape[0] * self.tokens, self.features)

    def forward_batch(self, X: FloatArray) -> FloatArray:
        R = self._rows(X)
        d = self.features
        mu = sum_features(R) / d
        c = R - mu
        var = sum_features(c * c) / d
        self._std = np.sqrt(var + self.epsilon)
        self._xhat = c / self._std
        self.A = (self.gamma * self._xhat + self.beta).reshape(X.shape)
        return self.A

    def _backward(self, downstream: FloatArray) -> None:
        self.delta_batch = downstream

    def downstream_batch(self) -> FloatArray:
        d = self.features
        dxhat = self._rows(self.delta_batch) * self.gamma
        m1 = sum_features(dxhat) / d
        m2 = sum_features(dxhat * self._xhat) / d
        return (((dxhat - m1) - self._xhat * m2) / self._std).reshape(self.delta_batch.shape)

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        delta = self._rows(self.delta_batch)
        self.grad_gamma += sum_rows(delta * self._xhat)
        self.grad_beta += sum_rows(delta)

    def reset_gradient_accum(self) -> None:
        self.grad_gamma = np.zeros(self.features)
        self.grad_beta = np.zeros(self.features)
