from __future__ import annotations

import indrajala_ml_array as pa

from indrajala_ml.model.rust_array_layer import RustArrayLayer


class CrossEntropyRustArrayLayer(RustArrayLayer):
    """
    The Rust-matmul-backed counterpart to CrossEntropyArrayLayer.
    Needs no new Rust primitive at all: `SoftmaxArrayLayer.compute_output_delta`'s own formula
    (`self.a - reference`) is algebraically identical to what cross-entropy needs, and its
    existing Rust-fused counterpart, `pa.layer_softmax_output_delta` (`fused.rs`), is already
    shape-agnostic (`require_same_shape` + elementwise subtract, no softmax-specific math) -
    checked directly against the Rust source, not assumed. `forward`/`forward_batch`/
    `compute_hidden_delta`/`compute_hidden_delta_batch`/`apply_accumulated_gradient` are all
    inherited unchanged from `RustArrayLayer`, the same "single sigmoid output needs nothing from
    any sibling" reasoning `CrossEntropyArrayLayer`'s own docstring gives.
    """

    def compute_output_delta(self, reference: "pa.Array") -> None:
        self.delta = pa.layer_softmax_output_delta(self.a, reference)

    def compute_output_delta_batch(self, reference_batch: "pa.Array") -> None:
        self.delta_batch = pa.layer_softmax_output_delta(self.A, reference_batch)
