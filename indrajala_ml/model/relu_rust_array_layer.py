from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.rust_array_layer import RustArrayLayer


class ReLURustArrayLayer(RustArrayLayer):
    """
    The Rust-matmul-backed counterpart to ReLUArrayLayer. Same forward/backward formulas (`max(0, z)`, a boolean-mask
    derivative), but each as a single fused Rust call (`layer_relu_forward`/
    `layer_relu_forward_batch`/`layer_relu_hidden_delta`/`layer_relu_hidden_delta_batch`,
    `fused.rs`, built on the Rust core's `array_relu`/`array_relu_mask` primitives) instead of a
    numpy expression - mirroring how `RustArrayLayer` itself relates to `ArrayLayer`.
    `apply_accumulated_gradient` is inherited unchanged from `RustArrayLayer`.

    Hidden-layer-only, matching `ReLUArrayLayer`'s own convention: `compute_output_delta`/
    `compute_output_delta_batch` raise `NotImplementedError`.
    """

    def forward(self, x: "pa.Array") -> "pa.Array":
        self.a = pa.layer_relu_forward(self.W, x, self.b)
        return self.a

    def forward_batch(self, X: "pa.Array") -> "pa.Array":
        self.A = pa.layer_relu_forward_batch(self.W, X, self.b)
        return self.A

    def compute_output_delta(self, reference: "pa.Array") -> None:
        raise NotImplementedError(
            "ReLURustArrayLayer is a hidden-layer activation, not an output one - an unbounded "
            "activation isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_output_delta_batch(self, reference_batch: "pa.Array") -> None:
        raise NotImplementedError(
            "ReLURustArrayLayer is a hidden-layer activation, not an output one - an unbounded "
            "activation isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_hidden_delta(self, next_layer: "RustArrayLayer") -> None:
        self.delta = pa.layer_relu_hidden_delta(next_layer.W, next_layer.delta, self.a)

    def compute_hidden_delta_batch(self, next_layer: "RustArrayLayer") -> None:
        self.delta_batch = pa.layer_relu_hidden_delta_batch(next_layer.W, next_layer.delta_batch, self.A)
