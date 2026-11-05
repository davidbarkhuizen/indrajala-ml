"""AffineOutputArrayLayer on the Rust backend: affine_forward*, its delta the given gradient."""

from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.rust.affine_rust_array_layer import AffineRustArrayLayer


class AffineOutputRustArrayLayer(AffineRustArrayLayer):
    """W x + b as the output layer: its delta is the given gradient at the output."""

    def compute_output_delta(self, reference: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.delta = reference

    def compute_output_delta_batch(self, reference_batch: pa.Array, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.delta_batch = reference_batch
