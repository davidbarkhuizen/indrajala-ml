from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.rust_array_layer import RustArrayLayer


class SoftmaxRustArrayLayer(RustArrayLayer):
    """
    SoftmaxArrayLayer on the Rust backend: forward* and compute_output_delta* are each one fused
    call (layer_softmax_*). Output only, size >= 2.
    """

    def __init__(self, size: int, input_size: int) -> None:
        assert size >= 2, f"a softmax layer needs at least 2 nodes to normalize over; got size={size}"
        super().__init__(size, input_size)

    def forward(self, x: pa.Array) -> pa.Array:
        self.a = pa.layer_softmax_forward(self.W, x, self.b)
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A = pa.layer_softmax_forward_batch(self.W, X, self.b)
        return self.A

    def compute_output_delta(self, reference: pa.Array) -> None:
        self.delta = pa.layer_softmax_output_delta(self.a, reference)

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        self.delta_batch = pa.layer_softmax_output_delta(self.A, reference_batch)
