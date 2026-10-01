# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, which strict mode takes for constants)
"""
A residual block's fork and add on the Rust backend (the residual-connections workplan, stage 4),
the counterparts of residual_array_layer.py's, with its formulas: the add's sum and the fork's are
Array additions. The affine layer that ends a body is affine_rust_array_layer.py's.

A dense layer's hidden delta is one fused call that reads the next layer's W and delta; before a
fork it is the fused skip op (D8), which reads the fork's body_first's W and delta and its add's
delta itself (rust_array_layer.py). So the fork computes its summed delta only when downstream*()
is asked for it: by an add before it (two blocks in a row), or batch norm's ReLU branch. A fused
predecessor pays no extra crossing.
"""

from __future__ import annotations

from typing import Any

import indrajala_math_rust as pa


class _ParameterFree:
    """What a fork and an add share: no parameters, so nothing to accumulate or step."""

    def compute_output_delta(self, reference: pa.Array) -> None:
        raise NotImplementedError(f"a {type(self).__name__} is hidden, inside a network")

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        raise NotImplementedError(f"a {type(self).__name__} is hidden, inside a network")

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        pass

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        pass


class ForkRustArrayLayer(_ParameterFree):
    """
    ForkArrayLayer on the Rust backend: forward passes its input on and keeps it. Its delta, the
    body's downstream plus the add's delta, is computed when downstream*() asks for it: a dense
    layer before it reads body_first and add itself, through the fused skip ops.
    """

    # set by the builder: the block's add, and its body's first layer
    add: AddRustArrayLayer
    body_first: Any

    def __init__(self, size: int) -> None:
        self.size = size
        self.input_size = size

    def forward(self, x: pa.Array) -> pa.Array:
        self.x = x
        return x

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.X = X
        return X

    def compute_hidden_delta(self, next_layer: Any) -> None:
        # lazy: downstream() sums the paths when a layer before asks for it
        pass

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        pass

    def downstream(self) -> pa.Array:
        return self.body_first.downstream() + self.add.delta

    def downstream_batch(self) -> pa.Array:
        return self.body_first.downstream_batch() + self.add.delta_batch


class AddRustArrayLayer(_ParameterFree):
    """AddArrayLayer on the Rust backend: y + x, its delta the next layer's downstream and its own downstream."""

    def __init__(self, fork: ForkRustArrayLayer) -> None:
        self.fork = fork
        self.size = fork.size
        self.input_size = fork.size

    def forward(self, x: pa.Array) -> pa.Array:
        self.a = x + self.fork.x
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A = X + self.fork.X
        return self.A

    def compute_hidden_delta(self, next_layer: Any) -> None:
        self.delta = next_layer.downstream()

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        self.delta_batch = next_layer.downstream_batch()

    def downstream(self) -> pa.Array:
        return self.delta

    def downstream_batch(self) -> pa.Array:
        return self.delta_batch
