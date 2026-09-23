from __future__ import annotations

import indrajala_math_rust as pa


def unfused_sgd_step(layer, input_activation: "pa.Array", learning_rate: float) -> None:
    """
    The sgd_step of a layer whose update isn't plain SGD (momentum, Adam, L2) or that isn't a
    dense layer at all (conv): accumulate_gradient then apply_accumulated_gradient at
    batch_size=1, exactly what RustArrayNetworkBase.learn called before sgd_step existed.
    """
    layer.accumulate_gradient(input_activation)
    layer.apply_accumulated_gradient(learning_rate, batch_size=1)


class RustArrayLayer:
    """
    The Rust-array-core-backed sibling of ArrayLayer. Every method here is a single fused Rust
    call (`fused.rs`) instead of a composition of individual `Array` operators, mirroring
    ArrayLayer's own method names/formulas exactly so this class is a drop-in swap for it.
    `indrajala_math_rust.Array` has no in-place arithmetic beyond `+=`/`-=`, so every method below
    rebinds `self.W`/`self.b`/`self._grad_W`/`self._grad_b` to the fused call's return value
    rather than mutating in place - the same rebind-not-mutate pattern `Array.__iadd__` itself
    already uses under the hood.
    """

    def __init__(self, size: int, input_size: int) -> None:
        self.size = size
        self.input_size = input_size

        self.W = pa.Array.zeros((size, input_size))
        self.b = pa.Array.zeros(size)

        self._grad_W = pa.Array.zeros((size, input_size))
        self._grad_b = pa.Array.zeros(size)

    def forward(self, x: "pa.Array") -> "pa.Array":
        self.a = pa.layer_forward(self.W, x, self.b)
        return self.a

    def forward_batch(self, X: "pa.Array") -> "pa.Array":
        self.A = pa.layer_forward_batch(self.W, X, self.b)
        return self.A

    def compute_output_delta(self, reference: "pa.Array") -> None:
        self.delta = pa.layer_output_delta(self.a, reference)

    def downstream(self) -> "pa.Array":
        # ArrayLayer.downstream: the gradient sent back to this layer's input, read by a conv or
        # pool layer before it. The dense hidden-delta methods keep their fused calls, which read
        # next_layer.W themselves - a dense layer is never followed by a conv or pool layer, and
        # splitting the fusion would add a boundary crossing to every dense backward step.
        return pa.layer_downstream(self.W, self.delta)

    def downstream_batch(self) -> "pa.Array":
        return pa.layer_downstream_batch(self.W, self.delta_batch)

    def compute_hidden_delta(self, next_layer: "RustArrayLayer") -> None:
        self.delta = pa.layer_hidden_delta(next_layer.W, next_layer.delta, self.a)

    def compute_output_delta_batch(self, reference_batch: "pa.Array") -> None:
        self.delta_batch = pa.layer_output_delta(self.A, reference_batch)

    def compute_hidden_delta_batch(self, next_layer: "RustArrayLayer") -> None:
        self.delta_batch = pa.layer_hidden_delta_batch(next_layer.W, next_layer.delta_batch, self.A)

    def accumulate_gradient(self, input_activation: "pa.Array") -> None:
        self._grad_W, self._grad_b = pa.layer_accumulate_gradient(
            self.delta, input_activation, self._grad_W, self._grad_b
        )

    def accumulate_gradient_batch(self, input_activation_batch: "pa.Array") -> None:
        self._grad_W, self._grad_b = pa.layer_accumulate_gradient_batch(
            self.delta_batch, input_activation_batch, self._grad_W, self._grad_b
        )

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.W, self.b = pa.layer_apply_accumulated_gradient(
            self.W, self.b, self._grad_W, self._grad_b, learning_rate, batch_size
        )
        self._reset_gradient_accum()

    def sgd_step(self, input_activation: "pa.Array", learning_rate: float) -> None:
        # accumulate_gradient then apply_accumulated_gradient(learning_rate, batch_size=1) as one
        # fused call, bit-identical to that pair. It relies on the accumulators being fresh
        # zeros, which they always are in RustArrayNetworkBase.learn, its only caller: apply
        # resets them after every step. The accumulators aren't touched, so they stay zero. A
        # subclass that overrides accumulate_gradient or apply_accumulated_gradient must also
        # override this (tests/test_rust_array_layer_sgd_step.py checks it).
        self.W, self.b = pa.layer_sgd_step(self.W, self.b, self.delta, input_activation, learning_rate)

    def _reset_gradient_accum(self) -> None:
        self._grad_W = pa.Array.zeros((self.size, self.input_size))
        self._grad_b = pa.Array.zeros(self.size)
