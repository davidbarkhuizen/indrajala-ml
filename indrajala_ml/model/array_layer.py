from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    The vectorized counterpart to backprop_node.sigmoid: the same 1/(1+e^-z) formula, but relying
    on numpy's own overflow behavior instead of a try/except - large negative z drives
    np.exp(-z) to inf, and 1/(1+inf) is 0.0 under IEEE 754, which is exactly the limiting value
    backprop_node.sigmoid's OverflowError branch returns by hand. Confirmed by
    tests/test_array_layer.py's dedicated overflow-boundary sweep, not assumed from the formulas
    looking equivalent.
    """

    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


def fan_in_aware_random_layer(size: int, previous_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    The fan-in-aware initialization draw (limit = 1/sqrt(fan_in)) shared by
    VectorizedMultiClassBackpropClassifierNetwork.randomize and
    ArrayBackpropClassifierNetwork.randomize - the array-level analogue of
    randomize_fan_in_aware (backprop_network_base.py), extracted here so both callers draw from
    one formula instead of two independent copies.
    """
    limit = 1.0 / np.sqrt(previous_size)
    W = np.random.uniform(-limit, limit, size=(size, previous_size))
    b = np.random.uniform(-limit, limit, size=(size,))
    return W, b


class ArrayLayer:
    """
    One backprop layer's weights/activations as whole arrays, not `size` separate BackpropNode
    objects. Provides both a single-example forward() and a batched forward_batch().
    """

    def __init__(self, size: int, input_size: int) -> None:
        self.size = size
        self.input_size = input_size

        self.W: np.ndarray = np.zeros((size, input_size))
        self.b: np.ndarray = np.zeros(size)

        # accumulated by accumulate_gradient(), consumed and reset by
        # apply_accumulated_gradient() - the array-valued analogue of BackpropNode's own
        # _weight_gradient_accum/_bias_gradient_accum pair
        self._grad_W: np.ndarray = np.zeros((size, input_size))
        self._grad_b: np.ndarray = np.zeros(size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = self.W @ x + self.b
        self.a = sigmoid(self.z)
        return self.a

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # X.shape == (batch_size, input_size); numpy broadcasts + self.b across every row, the
        # same formula as forward() applied to a whole batch at once instead of once per example
        self.Z = X @ self.W.T + self.b
        self.A = sigmoid(self.Z)
        return self.A

    def compute_output_delta(self, reference: np.ndarray) -> None:
        self.delta = (self.a - reference) * self.a * (1.0 - self.a)

    def compute_hidden_delta(self, next_layer: "ArrayLayer") -> None:
        # next_layer.W.T @ next_layer.delta replaces BackpropNode.compute_hidden_delta's
        # per-node Python sum() over downstream nodes with one matmul for the whole layer
        downstream = next_layer.W.T @ next_layer.delta
        self.delta = downstream * self.a * (1.0 - self.a)

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.delta_batch = (self.A - reference_batch) * self.A * (1.0 - self.A)

    def compute_hidden_delta_batch(self, next_layer: "ArrayLayer") -> None:
        # next_layer.delta_batch.shape == (batch_size, next_layer.size); next_layer.W.shape ==
        # (next_layer.size, self.size), so next_layer.delta_batch @ next_layer.W stacks
        # compute_hidden_delta's single-example downstream computation over every batch row
        downstream = next_layer.delta_batch @ next_layer.W
        self.delta_batch = downstream * self.A * (1.0 - self.A)

    def accumulate_gradient(self, input_activation: np.ndarray) -> None:
        # the single-example path - the same accumulate/apply split BackpropNode already uses,
        # one array op (an outer product) instead of a double Python loop over (node,
        # input_node) pairs. Callable repeatedly, once per example, before any weight is
        # written (see tests/test_array_layer.py's multi-example batch parity test).
        self._grad_W += np.outer(self.delta, input_activation)
        self._grad_b += self.delta

    def accumulate_gradient_batch(self, input_activation_batch: np.ndarray) -> None:
        # the one-shot batched path VectorizedMultiClassBackpropClassifierNetwork.learn_batch
        # uses instead of calling accumulate_gradient once per example:
        # self.delta_batch.T @ input_activation_batch computes the same sum of per-example outer
        # products as looping accumulate_gradient over every row, in one matrix multiply.
        self._grad_W += self.delta_batch.T @ input_activation_batch
        self._grad_b += self.delta_batch.sum(axis=0)

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.W -= learning_rate * self._grad_W / batch_size
        self.b -= learning_rate * self._grad_b / batch_size
        self._reset_gradient_accum()

    def _reset_gradient_accum(self) -> None:
        self._grad_W = np.zeros((self.size, self.input_size))
        self._grad_b = np.zeros(self.size)
