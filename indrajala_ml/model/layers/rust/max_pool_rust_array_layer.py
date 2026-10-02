from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.specs.hidden_layers import DeltaIsDownstream, Hidden, ParameterFree
from indrajala_ml.model.specs.window_geometry import pool_stride, validate_pool_arguments


class MaxPoolRustArrayLayer(Hidden[pa.Array], DeltaIsDownstream[pa.Array], ParameterFree[pa.Array]):
    """
    MaxPoolArrayLayer on the Rust backend: the same layouts and tie-breaking (the first maximal slot
    in row-major order), each method one fused call (conv.rs) with a pa.ConvGeometry whose
    kernel_size is pool_size.

    argmax_batch holds slot indices as floats (small exact integers), since
    indrajala_math_rust.Array has no integer type. No weights: the gradient methods are no-ops.
    Single-example calls pass 1D arrays to the batch ops, as ConvRustArrayLayer's do.
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        input_channels: int,
        pool_size: int,
        stride: int | None = None,
    ) -> None:

        stride = pool_stride(pool_size, stride)
        validate_pool_arguments(input_height, input_width, input_channels, pool_size, stride)
        self.geometry = pa.ConvGeometry(input_height, input_width, input_channels, pool_size, stride)

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.pool_size = pool_size
        self.stride = stride
        self.channel_count = input_channels

        self.out_height = self.geometry.out_height
        self.out_width = self.geometry.out_width

        self.input_size = self.geometry.input_size
        self.size = input_channels * self.geometry.positions

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A, self.argmax_batch = pa.max_pool_forward_batch(X, self.geometry)
        return self.A

    def forward(self, x: pa.Array) -> pa.Array:
        self.a, self.argmax = pa.max_pool_forward_batch(x, self.geometry)
        return self.a

    def downstream_batch(self) -> pa.Array:
        return pa.max_pool_downstream_batch(self.delta_batch, self.argmax_batch, self.geometry)

    def downstream(self) -> pa.Array:
        return pa.max_pool_downstream_batch(self.delta, self.argmax, self.geometry)
