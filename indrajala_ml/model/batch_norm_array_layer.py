# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, A, which strict mode takes for constants)
"""
Batch normalization of a linear layer, fused with its activation (the batch-norm workplan, D1): a
dense layer's each feature over the batch, a conv layer's each channel over the batch and every
position. Every expression is the README's (Batch normalization), in its grouping, and every sum
over the batch is a left fold in row order, the crate's, through np.cumsum: X.sum(axis=0) sums a
single feature pairwise (tests/test_summation_order.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import numpy as np

from indrajala_ml.model.array_layer import FloatArray, sigmoid
from indrajala_ml.model.layer_specs import ghost_groups, refuse_single_example

# a group's (m, d, var, std): its value count per feature, and what the backward pass reads
GroupStats = tuple[int, FloatArray, FloatArray, FloatArray]


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

    After a conv layer, positions is its out_height * out_width, and the size features are
    size // positions channels in the conv layer's channel-major layout. The statistics are
    computed on the (N * positions, channels) view, whose rows are the README's order: example by
    example, then position by position. The activations and deltas stay channel-major.

    With a group_size, a training batch's ghost groups (layer_specs.ghost_groups, D6) are each
    normalized as a batch of their own, and move the running averages in turn; the gradients of
    gamma and beta still sum over the whole batch.
    """

    decayed: ClassVar[tuple[bool, ...]] = (False, False)

    def __init__(
        self,
        size: int,
        activation: Literal["sigmoid", "relu"],
        epsilon: float,
        running_rate: float,
        positions: int = 1,
        group_size: int | None = None,
    ) -> None:
        assert positions >= 1 and size % positions == 0, f"{size} values aren't {positions} positions per channel"
        self.size = size
        self.input_size = size
        self.positions = positions
        features = size // positions
        self.activation = activation
        self.epsilon = epsilon
        self.running_rate = running_rate
        self.group_size = group_size
        self.training = False
        self._was_training = False

        # D5's initialization: nothing drawn
        self.gamma: FloatArray = np.ones(features)
        self.beta: FloatArray = np.zeros(features)
        self.running_mean: FloatArray = np.zeros(features)
        self.running_var: FloatArray = np.ones(features)

        self.grad_gamma: FloatArray = np.zeros(features)
        self.grad_beta: FloatArray = np.zeros(features)

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

    def _rows(self, X: FloatArray) -> FloatArray:
        # channel-major (N, C * P) as (N * P, C); a dense layer's X as it is
        if self.positions == 1:
            return X
        n = X.shape[0]
        return X.reshape(n, -1, self.positions).transpose(0, 2, 1).reshape(n * self.positions, -1)

    def _flat(self, R: FloatArray) -> FloatArray:
        # _rows' inverse
        if self.positions == 1:
            return R
        n = R.shape[0] // self.positions
        return R.reshape(n, self.positions, -1).transpose(0, 2, 1).reshape(n, self.size)

    def _activate(self, Y: FloatArray) -> FloatArray:
        return sigmoid(Y) if self.activation == "sigmoid" else np.maximum(0.0, Y)

    def _inference(self, X: FloatArray) -> FloatArray:
        xhat = (self._rows(X) - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self._flat(self._activate(self.gamma * xhat + self.beta))

    def forward(self, x: FloatArray) -> FloatArray:
        # classify_state's single-example forward pass: inference only
        if self.training:
            refuse_single_example(self)
        self.a = self._inference(x[np.newaxis, :])[0]
        return self.a

    def forward_batch(self, X: FloatArray) -> FloatArray:
        self._was_training = self.training
        if not self.training:
            self.A = self._inference(X)
            return self.A

        if X.shape[0] < 2:
            refuse_single_example(self)
        self._groups = ghost_groups(X.shape[0], self.group_size)
        # one group is the whole batch, without copies
        parts = [self._normalize(X[first:end] if len(self._groups) > 1 else X) for first, end in self._groups]
        self._stats = [stats for _A, _xhat, stats in parts]
        if len(parts) == 1:
            self.A, self._xhat, _stats = parts[0]
        else:
            self.A = np.concatenate([A for A, _xhat, _stats in parts])
            self._xhat = np.concatenate([xhat for _A, xhat, _stats in parts])
        return self.A

    def _normalize(self, X: FloatArray) -> tuple[FloatArray, FloatArray, GroupStats]:
        # one group's training forward pass, moving the running averages: its activations, its
        # x-hat as rows, and (m, d, var, std) for the backward pass
        R = self._rows(X)
        m = R.shape[0]
        mu = sum_rows(R) / m
        d = R - mu
        ss = sum_rows(d * d)
        var = ss / m
        std = np.sqrt(var + self.epsilon)
        xhat = d / std
        A = self._flat(self._activate(self.gamma * xhat + self.beta))

        rate = self.running_rate
        self.running_mean = (1 - rate) * self.running_mean + rate * mu
        self.running_var = (1 - rate) * self.running_var + rate * (ss / (m - 1))
        return A, xhat, (m, d, var, std)

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
        # dl/dx, the linear layer's delta: the paper's § 3 chain rule, term by term, per group
        if len(self._groups) == 1:
            return self._downstream(self.delta_batch, self._stats[0])
        return np.concatenate(
            [
                self._downstream(self.delta_batch[first:end], stats)
                for (first, end), stats in zip(self._groups, self._stats)
            ]
        )

    def _downstream(self, delta: FloatArray, stats: GroupStats) -> FloatArray:
        m, d, var, std = stats
        dxhat = self._rows(delta) * self.gamma
        inv_std = 1 / std
        inv_std3 = inv_std / (var + self.epsilon)
        dvar = sum_rows(dxhat * d * -0.5 * inv_std3)
        dmu = sum_rows(dxhat * -inv_std) + dvar * sum_rows(-2 * d) / m
        return self._flat(dxhat * inv_std + dvar * (2 * d) / m + dmu / m)

    def accumulate_gradient(self, input_activation: FloatArray) -> None:
        refuse_single_example(self)

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        delta = self._rows(self.delta_batch)
        self.grad_gamma += sum_rows(delta * self._xhat)
        self.grad_beta += sum_rows(delta)

    def reset_gradient_accum(self) -> None:
        self.grad_gamma = np.zeros(self.gamma.shape)
        self.grad_beta = np.zeros(self.beta.shape)
