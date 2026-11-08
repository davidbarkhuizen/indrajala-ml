# pyright: reportConstantRedefinition=false
# (Z, the batch's pre-activations, named as in the literature)
from __future__ import annotations

from typing import Any

import indrajala_math_rust as pa

from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.python.activations import HiddenActivation
from indrajala_ml.model.layers.rust.residual_rust_array_layer import ForkRustArrayLayer
from indrajala_ml.model.layers.rust.rust_array_layer import RustArrayLayer, reads_downstream


class ActivationRustArrayLayer(RustArrayLayer):
    """
    A hidden dense layer with a fused activation on the Rust backend, ReLU or GELU (the FFN
    activations workplan, D9): forward* and compute_hidden_delta* are each one call of the crate's
    activation family (layer_activation_*). The forward keeps z where the derivative reads it (the
    GELUs'), and the deltas read z, or a for ReLU (_cached). Hidden only: compute_output_delta*
    raise.
    """

    def __init__(self, size: int, input_size: int, activation: HiddenActivation) -> None:
        super().__init__(size, input_size)
        self.activation: HiddenActivation = activation
        self.z: pa.Array | None = None
        self.Z: pa.Array | None = None

    def forward(self, x: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        self.a, self.z = pa.layer_activation_forward(self.W, x, self.b, self.activation)
        return self.a

    def forward_batch(self, X: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> pa.Array:
        self.A, self.Z = pa.layer_activation_forward_batch(self.W, X, self.b, self.activation)
        return self.A

    def compute_output_delta(self, reference: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is a hidden-layer activation, not an output one - an unbounded "
            "activation isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_output_delta_batch(self, reference_batch: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is a hidden-layer activation, not an output one - an unbounded "
            "activation isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_hidden_delta(self, next_layer: Any, *, context: ForwardContext = NO_CONTEXT) -> None:
        cached = self.a if self.z is None else self.z
        if reads_downstream(next_layer):
            self.delta = pa.array_activation_mask(next_layer.downstream(context=context), cached, self.activation)
            return
        if isinstance(next_layer, ForkRustArrayLayer):
            # before a residual block: the fused skip op (D8, RustArrayLayer.compute_hidden_delta)
            body = next_layer.body_first
            self.delta = pa.layer_activation_hidden_delta_skip(
                body.W, body.delta, next_layer.add.delta, cached, self.activation
            )
            return
        self.delta = pa.layer_activation_hidden_delta(next_layer.W, next_layer.delta, cached, self.activation)

    def compute_hidden_delta_batch(self, next_layer: Any, *, context: ForwardContext = NO_CONTEXT) -> None:
        cached = self.A if self.Z is None else self.Z
        if reads_downstream(next_layer):
            downstream = next_layer.downstream_batch(context=context)
            self.delta_batch = pa.array_activation_mask(downstream, cached, self.activation)
            return
        if isinstance(next_layer, ForkRustArrayLayer):
            body = next_layer.body_first
            self.delta_batch = pa.layer_activation_hidden_delta_skip_batch(
                body.W, body.delta_batch, next_layer.add.delta_batch, cached, self.activation
            )
            return
        self.delta_batch = pa.layer_activation_hidden_delta_batch(
            next_layer.W, next_layer.delta_batch, cached, self.activation
        )


class ReLURustArrayLayer(ActivationRustArrayLayer):
    """ReLUArrayLayer on the Rust backend: the activation family's ReLU arm, which keeps no z."""

    def __init__(self, size: int, input_size: int) -> None:
        super().__init__(size, input_size, "relu")
