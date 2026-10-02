# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, which strict mode takes for constants)
from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.hidden_layers import DeltaIsDownstream, Hidden
from indrajala_ml.model.rust_array_layer import RustArrayLayer


class AffineRustArrayLayer(Hidden[pa.Array], DeltaIsDownstream[pa.Array], RustArrayLayer):
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
