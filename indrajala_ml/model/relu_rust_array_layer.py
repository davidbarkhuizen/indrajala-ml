from __future__ import annotations

from typing import Any

import indrajala_math_rust as pa

from indrajala_ml.model.residual_rust_array_layer import ForkRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer, before_layer_norm


class ReLURustArrayLayer(RustArrayLayer):
    """
    ReLUArrayLayer on the Rust backend: forward* and compute_hidden_delta* are each one fused call
    (layer_relu_*). Hidden only: compute_output_delta* raise.
    """

    def forward(self, x: pa.Array) -> pa.Array:
        self.a = pa.layer_relu_forward(self.W, x, self.b)
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A = pa.layer_relu_forward_batch(self.W, X, self.b)
        return self.A

    def compute_output_delta(self, reference: pa.Array) -> None:
        raise NotImplementedError(
            "ReLURustArrayLayer is a hidden-layer activation, not an output one - an unbounded "
            "activation isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        raise NotImplementedError(
            "ReLURustArrayLayer is a hidden-layer activation, not an output one - an unbounded "
            "activation isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_hidden_delta(self, next_layer: Any) -> None:
        if before_layer_norm(next_layer):
            self.delta = pa.array_relu_mask(next_layer.downstream(), self.a)
            return
        if isinstance(next_layer, ForkRustArrayLayer):
            # before a residual block: the fused skip op (D8, RustArrayLayer.compute_hidden_delta)
            body = next_layer.body_first
            self.delta = pa.layer_relu_hidden_delta_skip(body.W, body.delta, next_layer.add.delta, self.a)
            return
        self.delta = pa.layer_relu_hidden_delta(next_layer.W, next_layer.delta, self.a)

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        if before_layer_norm(next_layer):
            self.delta_batch = pa.array_relu_mask(next_layer.downstream_batch(), self.A)
            return
        if isinstance(next_layer, ForkRustArrayLayer):
            body = next_layer.body_first
            self.delta_batch = pa.layer_relu_hidden_delta_skip_batch(
                body.W, body.delta_batch, next_layer.add.delta_batch, self.A
            )
            return
        self.delta_batch = pa.layer_relu_hidden_delta_batch(next_layer.W, next_layer.delta_batch, self.A)
