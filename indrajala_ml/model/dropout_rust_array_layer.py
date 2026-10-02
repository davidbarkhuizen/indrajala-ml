from __future__ import annotations

from typing import Any

import indrajala_math_rust as pa

from indrajala_ml.model.residual_rust_array_layer import ForkRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer, before_layer_norm


class DropoutRustArrayLayer(RustArrayLayer):
    """
    DropoutArrayLayer on the Rust backend: forward*, compute_hidden_delta* are each one fused call
    (layer_dropout_*), drawing the mask from rng, the network's crate Generator. That is numpy's
    default_rng, so from generators in the same state the masks are the ones DropoutArrayLayer
    draws, and parity is exact in training too. drop_probability is required; a layer on its own
    draws from OS entropy.
    """

    def __init__(self, size: int, input_size: int, drop_probability: float) -> None:
        super().__init__(size, input_size)
        assert 0.0 <= drop_probability < 1.0, f"drop_probability must be in [0.0, 1.0); got {drop_probability}"
        self._drop_probability = drop_probability
        self._keep_probability = 1.0 - drop_probability
        self.training = False
        self.rng = pa.default_rng()

    def set_rng(self, rng: pa.Generator) -> None:
        self.rng = rng

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def forward(self, x: pa.Array) -> pa.Array:
        self.a, self._mask, self._base_activation = pa.layer_dropout_forward(
            self.W, x, self.b, self._drop_probability, self.training, self.rng
        )
        self._was_training = self.training
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A, self._mask_batch, self._base_activation_batch = pa.layer_dropout_forward_batch(
            self.W, X, self.b, self._drop_probability, self.training, self.rng
        )
        self._was_training = self.training
        return self.A

    def compute_hidden_delta(self, next_layer: Any) -> None:
        if before_layer_norm(next_layer):
            self.delta = pa.array_dropout_mask(
                next_layer.downstream(), self._base_activation, self._mask, self._keep_probability, self._was_training
            )
            return
        if isinstance(next_layer, ForkRustArrayLayer):
            # before a residual block: the fused skip op (D8, RustArrayLayer.compute_hidden_delta)
            body = next_layer.body_first
            self.delta = pa.layer_dropout_hidden_delta_skip(
                body.W,
                body.delta,
                next_layer.add.delta,
                self._base_activation,
                self._mask,
                self._keep_probability,
                self._was_training,
            )
            return
        self.delta = pa.layer_dropout_hidden_delta(
            next_layer.W,
            next_layer.delta,
            self._base_activation,
            self._mask,
            self._keep_probability,
            self._was_training,
        )

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        if before_layer_norm(next_layer):
            self.delta_batch = pa.array_dropout_mask(
                next_layer.downstream_batch(),
                self._base_activation_batch,
                self._mask_batch,
                self._keep_probability,
                self._was_training,
            )
            return
        if isinstance(next_layer, ForkRustArrayLayer):
            body = next_layer.body_first
            self.delta_batch = pa.layer_dropout_hidden_delta_skip_batch(
                body.W,
                body.delta_batch,
                next_layer.add.delta_batch,
                self._base_activation_batch,
                self._mask_batch,
                self._keep_probability,
                self._was_training,
            )
            return
        self.delta_batch = pa.layer_dropout_hidden_delta_batch(
            next_layer.W,
            next_layer.delta_batch,
            self._base_activation_batch,
            self._mask_batch,
            self._keep_probability,
            self._was_training,
        )
