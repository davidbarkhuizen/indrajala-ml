# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, which strict mode takes for constants)
from __future__ import annotations

from typing import Any

import indrajala_math_rust as pa

from indrajala_ml.model.rust_array_layer import RustArrayLayer


class AffineRustArrayLayer(RustArrayLayer):
    """
    AffineArrayLayer on the Rust backend: W x + b with no activation (affine_forward*), its delta
    the add's downstream. A RustArrayLayer, so its downstream, gradients and the optimizer's fused
    SGD step are a dense layer's. Hidden only.
    """

    def forward(self, x: pa.Array) -> pa.Array:
        self.a = pa.affine_forward(self.W, x, self.b)
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A = pa.affine_forward_batch(self.W, X, self.b)
        return self.A

    def compute_output_delta(self, reference: pa.Array) -> None:
        raise NotImplementedError("an affine layer is hidden, at the end of a residual block's body")

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        raise NotImplementedError("an affine layer is hidden, at the end of a residual block's body")

    def compute_hidden_delta(self, next_layer: Any) -> None:
        self.delta = next_layer.downstream()

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        self.delta_batch = next_layer.downstream_batch()
