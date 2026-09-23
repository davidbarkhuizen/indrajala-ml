from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer


class ReLUArrayLayer(ArrayLayer):
    """
    The array-based counterpart to relu_layer.ReLUNode/ReLULayer: max(0, z) instead of sigmoid,
    as whole-array numpy ops instead of a per-node Python loop.

    Unlike momentum/L2/Adam's own array siblings (which only touch apply_accumulated_gradient),
    ReLU changes the *activation* itself, so this overrides forward/forward_batch and
    compute_hidden_delta/compute_hidden_delta_batch instead - apply_accumulated_gradient is
    inherited unchanged from ArrayLayer, and there's no persistent per-parameter state at all.

    Hidden-layer-only by convention, matching ReLUNode's own posture: compute_output_delta/
    compute_output_delta_batch raise NotImplementedError rather than silently computing a
    meaningless sigmoid-shaped output delta on an unbounded ReLU activation - this class should
    never actually be used as an output layer.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = self.W @ x + self.b
        self.a = np.maximum(0.0, self.z)  # relu_layer.relu_activation, vectorized
        return self.a

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        self.Z = X @ self.W.T + self.b
        self.A = np.maximum(0.0, self.Z)
        return self.A

    def compute_output_delta(self, reference: np.ndarray) -> None:
        raise NotImplementedError(
            "ReLUArrayLayer is a hidden-layer activation, not an output one - an unbounded "
            "activation isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        raise NotImplementedError(
            "ReLUArrayLayer is a hidden-layer activation, not an output one - an unbounded "
            "activation isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_hidden_delta(self, next_layer: "ArrayLayer") -> None:
        downstream = next_layer.downstream()
        # relu_hidden_delta's derivative: 1 where z > 0 (equivalently a > 0), 0 otherwise - no
        # a*(1-a) damping term at all
        self.delta = downstream * (self.a > 0.0)

    def compute_hidden_delta_batch(self, next_layer: "ArrayLayer") -> None:
        downstream = next_layer.downstream_batch()
        self.delta_batch = downstream * (self.A > 0.0)
