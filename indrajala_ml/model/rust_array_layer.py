# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, A, which strict mode takes for constants)
from __future__ import annotations

import math

import indrajala_math_rust as pa

from indrajala_ml.model.array_parameters import WeightAndBias


def fan_in_aware_random_rust_layer(rng: pa.Generator, size: int, previous_size: int) -> tuple[pa.Array, pa.Array]:
    """
    fan_in_aware_random_layer drawn from the crate's Generator: the Rust backend's random_layer.
    The crate's Generator is numpy's default_rng (PCG64), so from generators in the same state
    this draws bit for bit what fan_in_aware_random_layer draws. math.sqrt, not ** 0.5: ** 0.5 is
    1 ULP off np.sqrt at some fan-ins.
    """
    limit = 1.0 / math.sqrt(previous_size)
    return fan_in_aware_random_rust_weights(rng, size, previous_size), rng.uniform(-limit, limit, size)


def fan_in_aware_random_rust_weights(rng: pa.Generator, size: int, previous_size: int) -> pa.Array:
    """fan_in_aware_random_rust_layer's W alone: the Rust backend's random_weights."""
    limit = 1.0 / math.sqrt(previous_size)
    return rng.uniform(-limit, limit, (size, previous_size))


class RustArrayLayer(WeightAndBias[pa.Array]):
    """
    ArrayLayer on the Rust backend, with the same methods and formulas, each one fused Rust call
    (fused.rs). indrajala_math_rust.Array has no in-place arithmetic beyond +=/-=, so each method
    rebinds W, b and the gradients to the call's result instead of mutating them.
    """

    def __init__(self, size: int, input_size: int) -> None:
        self.size = size
        self.input_size = input_size

        self.W = pa.Array.zeros((size, input_size))
        self.b = pa.Array.zeros(size)

        self.grad_W = pa.Array.zeros((size, input_size))
        self.grad_b = pa.Array.zeros(size)

    def forward(self, x: pa.Array) -> pa.Array:
        self.a = pa.layer_forward(self.W, x, self.b)
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A = pa.layer_forward_batch(self.W, X, self.b)
        return self.A

    def compute_output_delta(self, reference: pa.Array) -> None:
        self.delta = pa.layer_output_delta(self.a, reference)

    def downstream(self) -> pa.Array:
        # ArrayLayer.downstream: the gradient sent back to this layer's input, read by a conv or
        # pool layer before it. The dense hidden-delta methods keep their fused calls, which read
        # next_layer.W themselves - a dense layer is never followed by a conv or pool layer, and
        # splitting the fusion would add a boundary crossing to every dense backward step.
        return pa.layer_downstream(self.W, self.delta)

    def downstream_batch(self) -> pa.Array:
        return pa.layer_downstream_batch(self.W, self.delta_batch)

    def compute_hidden_delta(self, next_layer: RustArrayLayer) -> None:
        self.delta = pa.layer_hidden_delta(next_layer.W, next_layer.delta, self.a)

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        self.delta_batch = pa.layer_output_delta(self.A, reference_batch)

    def compute_hidden_delta_batch(self, next_layer: RustArrayLayer) -> None:
        self.delta_batch = pa.layer_hidden_delta_batch(next_layer.W, next_layer.delta_batch, self.A)

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        self.grad_W, self.grad_b = pa.layer_accumulate_gradient(self.delta, input_activation, self.grad_W, self.grad_b)

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        self.grad_W, self.grad_b = pa.layer_accumulate_gradient_batch(
            self.delta_batch, input_activation_batch, self.grad_W, self.grad_b
        )

    def reset_gradient_accum(self) -> None:
        self.grad_W = pa.Array.zeros((self.size, self.input_size))
        self.grad_b = pa.Array.zeros(self.size)
