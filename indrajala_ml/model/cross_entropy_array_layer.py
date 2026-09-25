from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer


class CrossEntropyArrayLayer(ArrayLayer):
    """
    A sigmoid output layer with binary cross-entropy loss: the delta is activation - target, without
    the a*(1-a) factor (CrossEntropyOutputNode over arrays). Every other method is ArrayLayer's.

    The delta is per node, so any size works: one node (as CrossEntropyOutputLayer) or class_count
    (one-vs-rest with cross-entropy, unlike softmax's joint normalization).
    """

    def compute_output_delta(self, reference: np.ndarray) -> None:
        self.delta = self.a - reference

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.delta_batch = self.A - reference_batch
