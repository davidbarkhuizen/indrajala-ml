# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, A, which strict mode takes for constants)
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from indrajala_ml.model.array_parameters import WeightAndBias

# the numpy backend's array: every numpy layer's weights, activations and gradients
FloatArray = npt.NDArray[np.float64]


def sigmoid(z: FloatArray) -> FloatArray:
    """
    backprop_node.sigmoid over arrays, 1/(1+e^-z). Large negative z overflows np.exp(-z) to inf,
    and 1/(1+inf) is 0.0, the value backprop_node.sigmoid returns on OverflowError
    (tests/test_array_layer.py sweeps the boundary).
    """

    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


def fan_in_aware_random_layer(size: int, previous_size: int) -> tuple[FloatArray, FloatArray]:
    """
    A layer's (W, b) drawn uniformly from [-limit, limit], limit = 1/sqrt(fan_in): the numpy
    backend's random_layer.
    """
    W = fan_in_aware_random_weights(size, previous_size)
    limit = 1.0 / np.sqrt(previous_size)
    b = np.random.uniform(-limit, limit, size=(size,))
    return W, b


def fan_in_aware_random_weights(size: int, previous_size: int) -> FloatArray:
    """fan_in_aware_random_layer's W alone: the numpy backend's random_weights."""
    limit = 1.0 / np.sqrt(previous_size)
    return np.random.uniform(-limit, limit, size=(size, previous_size))


class ArrayLayer(WeightAndBias[FloatArray]):
    """
    One sigmoid layer's weights and activations as arrays, not `size` BackpropNodes, with
    single-example (forward, delta, ...) and batch (forward_batch, delta_batch, ...) methods.
    The network's optimizer (optimizers.py) steps W and b from the accumulated gradients.
    """

    def __init__(self, size: int, input_size: int) -> None:
        self.size = size
        self.input_size = input_size

        self.W: FloatArray = np.zeros((size, input_size))
        self.b: FloatArray = np.zeros(size)

        # accumulated by accumulate_gradient*(), consumed by the network's optimizer, which then
        # calls reset_gradient_accum()
        self.grad_W: FloatArray = np.zeros((size, input_size))
        self.grad_b: FloatArray = np.zeros(size)

    def forward(self, x: FloatArray) -> FloatArray:
        self.z = self.W @ x + self.b
        self.a = sigmoid(self.z)
        return self.a

    def forward_batch(self, X: FloatArray) -> FloatArray:
        # X is (batch_size, input_size); self.b broadcasts across rows
        self.Z = X @ self.W.T + self.b
        self.A = sigmoid(self.Z)
        return self.A

    def compute_output_delta(self, reference: FloatArray) -> None:
        self.delta = (self.a - reference) * self.a * (1.0 - self.a)

    def downstream(self) -> FloatArray:
        # the gradient this layer sends back to its input, computed by the layer that owns the
        # weights (as BackpropLayer.downstream_sum), so a conv or pool layer can sit on either
        # side of any layer
        return self.W.T @ self.delta

    def downstream_batch(self) -> FloatArray:
        # downstream() for every row: (batch_size, size) @ (size, input_size)
        return self.delta_batch @ self.W

    def compute_hidden_delta(self, next_layer: ArrayLayer) -> None:
        downstream = next_layer.downstream()
        self.delta = downstream * self.a * (1.0 - self.a)

    def compute_output_delta_batch(self, reference_batch: FloatArray) -> None:
        self.delta_batch = (self.A - reference_batch) * self.A * (1.0 - self.A)

    def compute_hidden_delta_batch(self, next_layer: ArrayLayer) -> None:
        downstream = next_layer.downstream_batch()
        self.delta_batch = downstream * self.A * (1.0 - self.A)

    def accumulate_gradient(self, input_activation: FloatArray) -> None:
        # one example's gradient; callable once per example before any weight is written
        self.grad_W += np.outer(self.delta, input_activation)
        self.grad_b += self.delta

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        # the sum of every row's outer product, in one matrix multiply
        self.grad_W += self.delta_batch.T @ input_activation_batch
        self.grad_b += self.delta_batch.sum(axis=0)

    def reset_gradient_accum(self) -> None:
        self.grad_W = np.zeros((self.size, self.input_size))
        self.grad_b = np.zeros(self.size)
