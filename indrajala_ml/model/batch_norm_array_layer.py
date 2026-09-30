# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, A, which strict mode takes for constants)
"""
Batch normalization of a dense linear layer, each feature over the batch, fused with its
activation (the batch-norm workplan, D1). Every expression is the README's (Batch normalization),
in its grouping, and every sum over the batch is a left fold in row order, the crate's, through
np.cumsum: X.sum(axis=0) sums a single feature pairwise (tests/test_summation_order.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import numpy as np

from indrajala_ml.model.array_layer import FloatArray, sigmoid
from indrajala_ml.model.linear_array_layer import refuse_single_example


def sum_rows(values: FloatArray) -> FloatArray:
    """Each column's sum over the rows, a left fold in row order at any shape."""
    return np.cumsum(values, axis=0)[-1]


class BatchNormArrayLayer:
    """
    y = gamma * xhat + beta, then the activation, xhat normalized with the batch's statistics in
    training and the running averages in inference. gamma and beta are trained, without weight
    decay (D7); the running averages move in training forward passes only. Hidden only, and batch
    only in training (D4).

    training is set by set_training_mode. The backward pass reads _was_training, training as
    forward_batch saw it, since the network switches training off before the backward pass.
    """

    decayed: ClassVar[tuple[bool, ...]] = (False, False)

    def __init__(self, size: int, activation: Literal["sigmoid", "relu"], epsilon: float, running_rate: float) -> None:
        self.size = size
        self.input_size = size
        self.activation = activation
        self.epsilon = epsilon
        self.running_rate = running_rate
        self.training = False
        self._was_training = False

        # D5's initialization: nothing drawn
        self.gamma: FloatArray = np.ones(size)
        self.beta: FloatArray = np.zeros(size)
        self.running_mean: FloatArray = np.zeros(size)
        self.running_var: FloatArray = np.ones(size)

        self.grad_gamma: FloatArray = np.zeros(size)
        self.grad_beta: FloatArray = np.zeros(size)

    def parameters(self) -> tuple[FloatArray, ...]:
        return self.gamma, self.beta

    def gradients(self) -> tuple[FloatArray, ...]:
        return self.grad_gamma, self.grad_beta

    def set_parameters(self, parameters: Sequence[FloatArray]) -> None:
        self.gamma, self.beta = parameters

    def running_state(self) -> tuple[FloatArray, ...]:
        return self.running_mean, self.running_var

    def set_running_state(self, state: Sequence[FloatArray]) -> None:
        self.running_mean, self.running_var = state

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def _activate(self, Y: FloatArray) -> FloatArray:
        return sigmoid(Y) if self.activation == "sigmoid" else np.maximum(0.0, Y)

    def _inference(self, X: FloatArray) -> FloatArray:
        xhat = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self._activate(self.gamma * xhat + self.beta)

    def forward(self, x: FloatArray) -> FloatArray:
        # classify_state's single-example forward pass: inference only
        if self.training:
            refuse_single_example(self)
        self.a = self._inference(x)
        return self.a

    def forward_batch(self, X: FloatArray) -> FloatArray:
        self._was_training = self.training
        if not self.training:
            self.A = self._inference(X)
            return self.A

        m = X.shape[0]
        if m < 2:
            refuse_single_example(self)
        mu = sum_rows(X) / m
        d = X - mu
        var = sum_rows(d * d) / m
        std = np.sqrt(var + self.epsilon)
        self._xhat = d / std
        self.A = self._activate(self.gamma * self._xhat + self.beta)

        rate = self.running_rate
        self.running_mean = (1 - rate) * self.running_mean + rate * mu
        self.running_var = (1 - rate) * self.running_var + rate * (m / (m - 1) * var)

        self._m, self._d, self._std = m, d, std
        return self.A

    def compute_output_delta(self, reference: FloatArray) -> None:
        raise NotImplementedError("a batch-norm layer is hidden")

    def compute_output_delta_batch(self, reference_batch: FloatArray) -> None:
        raise NotImplementedError("a batch-norm layer is hidden")

    def compute_hidden_delta(self, next_layer: Any) -> None:
        refuse_single_example(self)

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        # dl/dy: the downstream times the activation's derivative, as ArrayLayer's and
        # ReLUArrayLayer's hidden deltas
        assert self._was_training, "the backward pass needs a training forward pass's batch statistics"
        downstream = next_layer.downstream_batch()
        if self.activation == "sigmoid":
            self.delta_batch = downstream * self.A * (1.0 - self.A)
        else:
            self.delta_batch = downstream * (self.A > 0.0)

    def downstream(self) -> FloatArray:
        refuse_single_example(self)

    def downstream_batch(self) -> FloatArray:
        # dl/dx, the linear layer's delta: the paper's § 3 chain rule, term by term
        m, d = self._m, self._d
        dxhat = self.delta_batch * self.gamma
        inv_std = 1 / self._std
        inv_std3 = inv_std * inv_std * inv_std
        dvar = sum_rows(dxhat * d * -0.5 * inv_std3)
        dmu = sum_rows(dxhat * -inv_std) + dvar * sum_rows(-2 * d) / m
        return dxhat * inv_std + dvar * (2 * d) / m + dmu / m

    def accumulate_gradient(self, input_activation: FloatArray) -> None:
        refuse_single_example(self)

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        self.grad_gamma += sum_rows(self.delta_batch * self._xhat)
        self.grad_beta += sum_rows(self.delta_batch)

    def reset_gradient_accum(self) -> None:
        self.grad_gamma = np.zeros(self.size)
        self.grad_beta = np.zeros(self.size)
