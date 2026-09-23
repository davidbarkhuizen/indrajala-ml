from __future__ import annotations

import indrajala_ml_array as pa

from indrajala_ml.model.rust_array_layer import RustArrayLayer


class SoftmaxRustArrayLayer(RustArrayLayer):
    """
    The Rust-matmul-backed counterpart to SoftmaxArrayLayer. Same forward/backward formulas (joint softmax normalization,
    `activation - target` delta with no `a*(1-a)` term), but each as a single fused Rust call
    (`layer_softmax_forward`/`layer_softmax_forward_batch`/`layer_softmax_output_delta`,
    `fused.rs`, built on the Rust core's `array_softmax` primitive) instead of a numpy expression -
    mirroring how `RustArrayLayer` itself relates to `ArrayLayer`. `compute_hidden_delta`/
    `compute_hidden_delta_batch`/`apply_accumulated_gradient` are inherited unchanged from
    `RustArrayLayer` - softmax's cross-node coupling only affects the forward pass.

    Output-layer-only, matching `SoftmaxArrayLayer`'s own `size >= 2` convention.
    """

    def __init__(self, size: int, input_size: int) -> None:
        assert size >= 2, f"a softmax layer needs at least 2 nodes to normalize over; got size={size}"
        super().__init__(size, input_size)

    def forward(self, x: "pa.Array") -> "pa.Array":
        self.a = pa.layer_softmax_forward(self.W, x, self.b)
        return self.a

    def forward_batch(self, X: "pa.Array") -> "pa.Array":
        self.A = pa.layer_softmax_forward_batch(self.W, X, self.b)
        return self.A

    def compute_output_delta(self, reference: "pa.Array") -> None:
        self.delta = pa.layer_softmax_output_delta(self.a, reference)

    def compute_output_delta_batch(self, reference_batch: "pa.Array") -> None:
        self.delta_batch = pa.layer_softmax_output_delta(self.A, reference_batch)
