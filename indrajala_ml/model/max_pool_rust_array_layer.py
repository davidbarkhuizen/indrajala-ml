from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.max_pool_array_layer import validate_pool_arguments


class MaxPoolRustArrayLayer:
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

        stride = pool_size if stride is None else stride
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

    def compute_output_delta(self, reference: pa.Array) -> None:
        raise NotImplementedError("MaxPoolRustArrayLayer is a hidden layer, not an output one.")

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        self.compute_output_delta(reference_batch)

    def compute_hidden_delta_batch(self, next_layer) -> None:
        # max is the identity on its winning input - no activation derivative to multiply in
        self.delta_batch = next_layer.downstream_batch()

    def compute_hidden_delta(self, next_layer) -> None:
        self.delta = next_layer.downstream()

    def downstream_batch(self) -> pa.Array:
        return pa.max_pool_downstream_batch(self.delta_batch, self.argmax_batch, self.geometry)

    def downstream(self) -> pa.Array:
        return pa.max_pool_downstream_batch(self.delta, self.argmax, self.geometry)

    # weight-free: every gradient hook below is a deliberate no-op

    def accumulate_gradient_batch(self, _input_activation_batch: pa.Array) -> None:
        pass

    def accumulate_gradient(self, _input_activation: pa.Array) -> None:
        pass

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        pass

    def sgd_step(self, _input_activation: pa.Array, learning_rate: float) -> None:
        pass
