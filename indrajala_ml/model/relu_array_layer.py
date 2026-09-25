from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer


class ReLUArrayLayer(ArrayLayer):
    """
    ReLU (max(0, z)) hidden layer over arrays, as ReLUNode/ReLULayer. ReLU changes the activation,
    so this overrides forward* and compute_hidden_delta*; the update is ArrayLayer's. Hidden only:
    compute_output_delta* raise.
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
        # derivative 1 where a > 0 (z > 0), else 0
        self.delta = downstream * (self.a > 0.0)

    def compute_hidden_delta_batch(self, next_layer: "ArrayLayer") -> None:
        downstream = next_layer.downstream_batch()
        self.delta_batch = downstream * (self.A > 0.0)
