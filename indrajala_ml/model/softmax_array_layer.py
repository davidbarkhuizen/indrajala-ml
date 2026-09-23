from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer


class SoftmaxArrayLayer(ArrayLayer):
    """
    The array-based counterpart to softmax_output_layer.SoftmaxOutputNode/SoftmaxOutputLayer:
    joint softmax normalization across the whole output vector instead of an independent per-node
    sigmoid, as whole-array numpy ops instead of a per-node Python loop.

    Output-layer-only, matching SoftmaxOutputLayer's own `assert size >= 2` convention (a
    single-node softmax has nothing to normalize against) - unlike ReLU's own hidden-layer-only
    array sibling, this overrides forward/forward_batch and compute_output_delta/
    compute_output_delta_batch, not compute_hidden_delta: softmax's cross-node coupling only
    affects the forward pass, so whatever layer feeds this one still calls the inherited,
    unmodified compute_hidden_delta/compute_hidden_delta_batch, which only ever calls
    next_layer.downstream()/downstream_batch() (next_layer.W/next_layer.delta), never reads
    next_layer.a directly.
    """

    def __init__(self, size: int, input_size: int) -> None:
        assert size >= 2, f"a softmax layer needs at least 2 nodes to normalize over; got size={size}"
        super().__init__(size, input_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = self.W @ x + self.b
        shifted = self.z - np.max(self.z)  # same numerically-stable shift as SoftmaxOutputLayer.forward
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
        self.delta = self.a - reference  # SoftmaxOutputNode's own simplification, no a*(1-a) term

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.delta_batch = self.A - reference_batch
