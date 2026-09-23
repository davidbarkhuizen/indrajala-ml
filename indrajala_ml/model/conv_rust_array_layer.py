from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.conv_array_layer import validate_conv_arguments


class ConvRustArrayLayer:
    """
    The Rust-array-core-backed counterpart to ConvArrayLayer: the same ReLU convolutional hidden
    layer, 'valid' padding, with the same layouts (see ConvArrayLayer's docstring) - channel-major
    (N, C*H*W) activations, a (channel_count, C*k*k) kernel matrix, and (N*P, C*k*k) im2col
    columns - but each method a single fused Rust call (`conv.rs`). Every conv tensor crosses the
    boundary as a matrix: indrajala_math_rust.Array stays 1D/2D.

    Arguments are validated as ConvArrayLayer validates them; the shape arithmetic then lives in
    one pa.ConvGeometry, built here and passed to every call.

    The Rust conv ops are batch-only. The single-example path reshapes x/delta to (1, n) and
    calls them, the same N = 1 wrapping ConvArrayLayer uses (the reshape copies).
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        input_channels: int,
        kernel_size: int,
        channel_count: int,
        stride: int = 1,
    ) -> None:

        validate_conv_arguments(input_height, input_width, input_channels, kernel_size, channel_count, stride)
        self.geometry = pa.ConvGeometry(input_height, input_width, input_channels, kernel_size, stride)

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.channel_count = channel_count
        self.stride = stride

        self.out_height = self.geometry.out_height
        self.out_width = self.geometry.out_width
        self.positions = self.geometry.positions

        self.fan_in = self.geometry.fan_in
        self.input_size = self.geometry.input_size
        self.size = channel_count * self.positions

        self.W = pa.Array.zeros((channel_count, self.fan_in))
        self.b = pa.Array.zeros(channel_count)

        self._grad_W = pa.Array.zeros((channel_count, self.fan_in))
        self._grad_b = pa.Array.zeros(channel_count)

    def forward_batch(self, X: "pa.Array") -> "pa.Array":
        self.Z, self.A, self._cols = pa.conv_forward_batch(self.W, X, self.b, self.geometry)
        return self.A

    def forward(self, x: "pa.Array") -> "pa.Array":
        self.forward_batch(x.reshape((1, self.input_size)))
        self.a = self.A.reshape(self.size)
        self.z = self.Z.reshape(self.size)
        return self.a

    def compute_output_delta(self, reference: "pa.Array") -> None:
        raise NotImplementedError(
            "ConvRustArrayLayer is a hidden layer, not an output one - see ConvUnit's identical "
            "guard: an unbounded ReLU activation isn't suited to any of this codebase's "
            "output-layer contracts."
        )

    def compute_output_delta_batch(self, reference_batch: "pa.Array") -> None:
        self.compute_output_delta(reference_batch)

    def compute_hidden_delta_batch(self, next_layer) -> None:
        # array_relu_mask: derivative 0 at exactly z == 0, the same convention as ConvArrayLayer
        self.delta_batch = pa.array_relu_mask(next_layer.downstream_batch(), self.A)

    def compute_hidden_delta(self, next_layer) -> None:
        self.delta = pa.array_relu_mask(next_layer.downstream(), self.a)

    def downstream_batch(self) -> "pa.Array":
        return pa.conv_downstream_batch(self.W, self.delta_batch, self.geometry)

    def downstream(self) -> "pa.Array":
        dX = pa.conv_downstream_batch(self.W, self.delta.reshape((1, self.size)), self.geometry)
        return dX.reshape(self.input_size)

    def accumulate_gradient_batch(self, _input_activation_batch: "pa.Array") -> None:
        # reads the im2col columns forward cached, as ConvArrayLayer._accumulate does
        self._grad_W, self._grad_b = pa.conv_accumulate_gradient_batch(
            self.delta_batch, self._cols, self._grad_W, self._grad_b, self.geometry
        )

    def accumulate_gradient(self, _input_activation: "pa.Array") -> None:
        self._grad_W, self._grad_b = pa.conv_accumulate_gradient_batch(
            self.delta.reshape((1, self.size)), self._cols, self._grad_W, self._grad_b, self.geometry
        )

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.W, self.b = pa.layer_apply_accumulated_gradient(
            self.W, self.b, self._grad_W, self._grad_b, learning_rate, batch_size
        )
        self._reset_gradient_accum()

    def _reset_gradient_accum(self) -> None:
        self._grad_W = pa.Array.zeros((self.channel_count, self.fan_in))
        self._grad_b = pa.Array.zeros(self.channel_count)
