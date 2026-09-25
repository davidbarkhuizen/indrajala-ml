from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    backprop_node.sigmoid over arrays, 1/(1+e^-z). Large negative z overflows np.exp(-z) to inf,
    and 1/(1+inf) is 0.0, the value backprop_node.sigmoid returns on OverflowError
    (tests/test_array_layer.py sweeps the boundary).
    """

    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


def unfused_sgd_step(layer, input_activation, learning_rate: float) -> None:
    """
    accumulate_gradient then apply_accumulated_gradient at batch_size=1: the sgd_step of every numpy
    layer, and of the Rust layers with no fused step (momentum, Adam, L2, conv).
    """
    layer.accumulate_gradient(input_activation)
    layer.apply_accumulated_gradient(learning_rate, batch_size=1)


def fan_in_aware_random_layer(size: int, previous_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    A layer's (W, b) drawn uniformly from [-limit, limit], limit = 1/sqrt(fan_in): the numpy
    backend's random_layer.
    """
    limit = 1.0 / np.sqrt(previous_size)
    W = np.random.uniform(-limit, limit, size=(size, previous_size))
    b = np.random.uniform(-limit, limit, size=(size,))
    return W, b


class ArrayLayer:
    """
    One sigmoid layer's weights and activations as arrays, not `size` BackpropNodes, with
    single-example (forward, delta, ...) and batch (forward_batch, delta_batch, ...) methods.
    """

    # the constructor keyword arguments after (size, input_size) that a subclass takes (e.g.
    # MomentumArrayLayer's momentum); ArrayNetworkBase passes them from the network's attributes
    # of the same names
    hyperparameters: tuple[str, ...] = ()

    def __init__(self, size: int, input_size: int) -> None:
        self.size = size
        self.input_size = input_size

        self.W: np.ndarray = np.zeros((size, input_size))
        self.b: np.ndarray = np.zeros(size)

        # accumulated by accumulate_gradient*(), consumed and reset by
        # apply_accumulated_gradient()
        self._grad_W: np.ndarray = np.zeros((size, input_size))
        self._grad_b: np.ndarray = np.zeros(size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = self.W @ x + self.b
        self.a = sigmoid(self.z)
        return self.a

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # X is (batch_size, input_size); self.b broadcasts across rows
        self.Z = X @ self.W.T + self.b
        self.A = sigmoid(self.Z)
        return self.A

    def compute_output_delta(self, reference: np.ndarray) -> None:
        self.delta = (self.a - reference) * self.a * (1.0 - self.a)

    def downstream(self) -> np.ndarray:
        # the gradient this layer sends back to its input, computed by the layer that owns the
        # weights (as BackpropLayer.downstream_sum), so a conv or pool layer can sit on either
        # side of any layer
        return self.W.T @ self.delta

    def downstream_batch(self) -> np.ndarray:
        # downstream() for every row: (batch_size, size) @ (size, input_size)
        return self.delta_batch @ self.W

    def compute_hidden_delta(self, next_layer: "ArrayLayer") -> None:
        downstream = next_layer.downstream()
        self.delta = downstream * self.a * (1.0 - self.a)

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.delta_batch = (self.A - reference_batch) * self.A * (1.0 - self.A)

    def compute_hidden_delta_batch(self, next_layer: "ArrayLayer") -> None:
        downstream = next_layer.downstream_batch()
        self.delta_batch = downstream * self.A * (1.0 - self.A)

    def accumulate_gradient(self, input_activation: np.ndarray) -> None:
        # one example's gradient; callable once per example before any weight is written
        self._grad_W += np.outer(self.delta, input_activation)
        self._grad_b += self.delta

    def accumulate_gradient_batch(self, input_activation_batch: np.ndarray) -> None:
        # the sum of every row's outer product, in one matrix multiply
        self._grad_W += self.delta_batch.T @ input_activation_batch
        self._grad_b += self.delta_batch.sum(axis=0)

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.W -= learning_rate * self._grad_W / batch_size
        self.b -= learning_rate * self._grad_b / batch_size
        self._reset_gradient_accum()

    def sgd_step(self, input_activation: np.ndarray, learning_rate: float) -> None:
        # the single-example step ArrayNetworkBase._learn_input calls; RustArrayLayer fuses it
        unfused_sgd_step(self, input_activation, learning_rate)

    def _reset_gradient_accum(self) -> None:
        self._grad_W = np.zeros((self.size, self.input_size))
        self._grad_b = np.zeros(self.size)
