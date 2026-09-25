from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.rust_array_layer import RustArrayLayer


class DropoutRustArrayLayer(RustArrayLayer):
    """
    DropoutArrayLayer on the Rust backend: forward*, compute_hidden_delta* are each one fused call
    (layer_dropout_*), drawing the mask with the crate's bernoulli_mask. The crate's RNG is numpy's
    np.random in a separate state, so after pa.seed(s) the masks are the ones DropoutArrayLayer
    draws after np.random.seed(s), and parity is exact in training too. drop_probability is
    required.
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

    def forward(self, x: pa.Array) -> pa.Array:
        self.a, self._mask, self._base_activation = pa.layer_dropout_forward(
            self.W, x, self.b, self._drop_probability, self.training
        )
        self._was_training = self.training
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A, self._mask_batch, self._base_activation_batch = pa.layer_dropout_forward_batch(
            self.W, X, self.b, self._drop_probability, self.training
        )
        self._was_training = self.training
        return self.A

    def compute_hidden_delta(self, next_layer: RustArrayLayer) -> None:
        self.delta = pa.layer_dropout_hidden_delta(
            next_layer.W,
            next_layer.delta,
            self._base_activation,
            self._mask,
            self._keep_probability,
            self._was_training,
        )

    def compute_hidden_delta_batch(self, next_layer: RustArrayLayer) -> None:
        self.delta_batch = pa.layer_dropout_hidden_delta_batch(
            next_layer.W,
            next_layer.delta_batch,
            self._base_activation_batch,
            self._mask_batch,
            self._keep_probability,
            self._was_training,
        )
