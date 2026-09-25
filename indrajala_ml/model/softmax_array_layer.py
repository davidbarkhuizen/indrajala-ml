from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer


class SoftmaxArrayLayer(ArrayLayer):
    """
    Softmax output layer over arrays, as SoftmaxOutputNode/SoftmaxOutputLayer: activations
    normalized jointly across the layer. Output only, size >= 2. Softmax couples the nodes in the
    forward pass only, so this overrides forward* and compute_output_delta*; the layer before it
    uses the inherited downstream().
    """

    def __init__(self, size: int, input_size: int) -> None:
        assert size >= 2, f"a softmax layer needs at least 2 nodes to normalize over; got size={size}"
        super().__init__(size, input_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = self.W @ x + self.b
        shifted = self.z - np.max(self.z)  # the stable shift of SoftmaxOutputLayer.forward
        exp_values = np.exp(shifted)
        self.a = exp_values / exp_values.sum()
        return self.a

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        self.Z = X @ self.W.T + self.b
        shifted = self.Z - self.Z.max(axis=1, keepdims=True)  # row-wise max, one row per example
        exp_values = np.exp(shifted)
        self.A = exp_values / exp_values.sum(axis=1, keepdims=True)
        return self.A

    def compute_output_delta(self, reference: np.ndarray) -> None:
        self.delta = self.a - reference  # no a*(1-a) term, as SoftmaxOutputNode

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.delta_batch = self.A - reference_batch
