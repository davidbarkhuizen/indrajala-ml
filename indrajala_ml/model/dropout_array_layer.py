from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer, sigmoid


class DropoutArrayLayer(ArrayLayer):
    """
    The array-based counterpart to dropout_layer.make_dropout_node_cls/make_dropout_layer_cls:
    the same inverted-dropout mechanism (Srivastava et al., 2014) - a training-time-only,
    per-forward-pass random mask zeroing a fraction of this layer's activations, rescaling the
    kept ones by 1/keep_probability - but as whole-array numpy ops over the layer's (size,
    input_size) weight matrix, instead of a per-node Python loop.

    Unlike momentum/L2/Adam's own array siblings (which only touch apply_accumulated_gradient),
    dropout changes the *activation* itself, so this overrides forward/forward_batch and
    compute_hidden_delta/compute_hidden_delta_batch instead - accumulate_gradient/
    apply_accumulated_gradient are inherited unchanged from ArrayLayer, and there's no
    persistent per-parameter state at all (training/_mask/_base_activation are all
    per-forward-pass-scoped, the same category _activation/delta already are).

    training is a plain mutable attribute (default False, the same safe-failure-mode default
    dropout_layer.py's own DropoutNode chooses), toggled by set_training_mode - the array-level
    counterpart to DropoutLayer.set_training_mode. The backward pass must read
    _base_activation (the *pre*-mask sigmoid, not self.a/self.A) and a forward-time snapshot of
    training (_was_training, not the live attribute) - the same two subtleties
    dropout_layer.py's own DropoutNode design found by testing, re-derived here rather than
    assumed to carry over unchanged.

    drop_probability is a required constructor argument, no default - the same posture
    make_dropout_node_cls itself takes.
    """

    def __init__(self, size: int, input_size: int, drop_probability: float) -> None:
        super().__init__(size, input_size)
        assert 0.0 <= drop_probability < 1.0, f"drop_probability must be in [0.0, 1.0); got {drop_probability}"
        self._drop_probability = drop_probability
        self._keep_probability = 1.0 - drop_probability
        self.training = False

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = self.W @ x + self.b
        base = sigmoid(self.z)
        if self.training:
            self._mask = (np.random.random(self.size) >= self._drop_probability).astype(np.float64)
            self.a = base * self._mask / self._keep_probability
        else:
            self._mask = np.ones(self.size)
            self.a = base
        self._base_activation = base
        self._was_training = self.training
        return self.a

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # one independent mask row per example (batch_size, self.size), not one shared mask for
        # the whole batch - dropout's whole point is a fresh, independent draw per forward pass,
        # and a batched forward pass is still batch_size independent forward passes from
        # dropout's perspective
        batch_size = X.shape[0]
        self.Z = X @ self.W.T + self.b
        base = sigmoid(self.Z)
        if self.training:
            self._mask_batch = (
                np.random.random((batch_size, self.size)) >= self._drop_probability
            ).astype(np.float64)
            self.A = base * self._mask_batch / self._keep_probability
        else:
            self._mask_batch = np.ones((batch_size, self.size))
            self.A = base
        self._base_activation_batch = base
        self._was_training = self.training
        return self.A

    def compute_hidden_delta(self, next_layer: "ArrayLayer") -> None:
        downstream = next_layer.W.T @ next_layer.delta
        sigmoid_derivative = self._base_activation * (1.0 - self._base_activation)
        scale = (self._mask / self._keep_probability) if self._was_training else 1.0
        self.delta = downstream * sigmoid_derivative * scale

    def compute_hidden_delta_batch(self, next_layer: "ArrayLayer") -> None:
        downstream = next_layer.delta_batch @ next_layer.W
        sigmoid_derivative = self._base_activation_batch * (1.0 - self._base_activation_batch)
        scale = (self._mask_batch / self._keep_probability) if self._was_training else 1.0
        self.delta_batch = downstream * sigmoid_derivative * scale
