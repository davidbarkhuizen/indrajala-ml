from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.rust_array_layer import RustArrayLayer


class CrossEntropyRustArrayLayer(RustArrayLayer):
    """
    CrossEntropyArrayLayer on the Rust backend. Its delta, activation - target, is what
    pa.layer_softmax_output_delta computes: that op is an elementwise subtract with no
    softmax-specific maths, so no new Rust op is needed.
    """

    def compute_output_delta(self, reference: pa.Array) -> None:
        self.delta = pa.layer_softmax_output_delta(self.a, reference)

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        self.delta_batch = pa.layer_softmax_output_delta(self.A, reference_batch)
