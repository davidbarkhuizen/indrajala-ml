# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, which strict mode takes for constants)
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import indrajala_math_rust as pa

from indrajala_ml.model.layer_specs import refuse_single_example


class LinearRustArrayLayer:
    """
    LinearArrayLayer on the Rust backend, with the same methods and formulas, each one Rust call:
    Z = X @ W.T with no bias and no activation, before a batch-norm layer. Hidden only, and batch
    only in training.
    """

    decayed: ClassVar[tuple[bool, ...]] = (True,)

    def __init__(self, size: int, input_size: int) -> None:
        self.size = size
        self.input_size = input_size

        self.W = pa.Array.zeros((size, input_size))
        self.grad_W = pa.Array.zeros((size, input_size))

    def parameters(self) -> tuple[pa.Array, ...]:
        return (self.W,)

    def gradients(self) -> tuple[pa.Array, ...]:
        return (self.grad_W,)

    def set_parameters(self, parameters: Sequence[pa.Array]) -> None:
        (self.W,) = parameters

    def forward(self, x: pa.Array) -> pa.Array:
        # classify_state's single-example forward pass
        self.a = pa.linear_forward(self.W, x)
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A = pa.linear_forward_batch(self.W, X)
        return self.A

    def compute_output_delta(self, reference: pa.Array) -> None:
        raise NotImplementedError("a linear layer is hidden, before a batch-norm layer")

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        raise NotImplementedError("a linear layer is hidden, before a batch-norm layer")

    def compute_hidden_delta(self, next_layer: Any) -> None:
        refuse_single_example(self)

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        self.delta_batch = next_layer.downstream_batch()

    def downstream(self) -> pa.Array:
        refuse_single_example(self)

    def downstream_batch(self) -> pa.Array:
        # RustArrayLayer.downstream_batch: the gradient into a conv or pool layer before this one
        return pa.layer_downstream_batch(self.W, self.delta_batch)

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        refuse_single_example(self)

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        self.grad_W = pa.linear_accumulate_gradient_batch(self.delta_batch, input_activation_batch, self.grad_W)

    def reset_gradient_accum(self) -> None:
        self.grad_W = pa.Array.zeros((self.size, self.input_size))
