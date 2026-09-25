from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer, sigmoid


class DropoutArrayLayer(ArrayLayer):
    """
    Inverted dropout (Srivastava et al., 2014) over arrays, as make_dropout_node_cls: in training
    each forward pass zeroes a random fraction of activations and scales the rest by
    1/keep_probability. Dropout changes the activation, so this overrides forward* and
    compute_hidden_delta*; the update is ArrayLayer's.

    training defaults to False and is set by set_training_mode. The backward pass reads
    _base_activation (the sigmoid before the mask, not self.a) and _was_training (training as
    forward saw it, since the network switches training off before the backward pass).
    drop_probability is required.
    """

    hyperparameters = ("drop_probability",)

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
        # an independent mask row per example, as batch_size single forward passes would draw
        batch_size = X.shape[0]
        self.Z = X @ self.W.T + self.b
        base = sigmoid(self.Z)
        if self.training:
            self._mask_batch = (np.random.random((batch_size, self.size)) >= self._drop_probability).astype(np.float64)
            self.A = base * self._mask_batch / self._keep_probability
        else:
            self._mask_batch = np.ones((batch_size, self.size))
            self.A = base
        self._base_activation_batch = base
        self._was_training = self.training
        return self.A

    def compute_hidden_delta(self, next_layer: ArrayLayer) -> None:
        downstream = next_layer.downstream()
        sigmoid_derivative = self._base_activation * (1.0 - self._base_activation)
        scale = (self._mask / self._keep_probability) if self._was_training else 1.0
        self.delta = downstream * sigmoid_derivative * scale

    def compute_hidden_delta_batch(self, next_layer: ArrayLayer) -> None:
        downstream = next_layer.downstream_batch()
        sigmoid_derivative = self._base_activation_batch * (1.0 - self._base_activation_batch)
        scale = (self._mask_batch / self._keep_probability) if self._was_training else 1.0
        self.delta_batch = downstream * sigmoid_derivative * scale
