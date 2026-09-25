# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, A, which strict mode takes for constants)
from __future__ import annotations

from typing import ClassVar, cast

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from indrajala_ml.model.array_layer import FloatArray, unfused_sgd_step
from indrajala_ml.model.array_protocols import ArrayNetworkLayer


def validate_conv_arguments(
    input_height: int, input_width: int, input_channels: int, kernel_size: int, channel_count: int, stride: int
) -> None:
    # ConvLayer's constructor checks, less the input_layer node count; shared with
    # ConvRustArrayLayer
    assert input_channels >= 1, f"input_channels must be at least 1; got {input_channels}"
    assert kernel_size >= 1, f"kernel_size must be at least 1; got {kernel_size}"
    assert channel_count >= 1, f"channel_count must be at least 1; got {channel_count}"
    assert stride >= 1, f"stride must be at least 1; got {stride}"
    assert kernel_size <= input_height and kernel_size <= input_width, (
        f"kernel_size ({kernel_size}) must fit within input_height x input_width ({input_height}x{input_width})"
    )


class ConvArrayLayer:
    """
    ConvLayer (conv_layer.py) over numpy arrays: a ReLU convolutional hidden layer, 'valid'
    padding, with the kernels as one matrix and a batch's receptive fields as one im2col array.

    Not an ArrayLayer: size is the flattened output count (channel_count * out_height * out_width),
    while W is (channel_count, input_channels * kernel_size**2). It implements the methods
    ArrayNetworkBase calls (forward*, compute_*_delta*, downstream*, accumulate_gradient*,
    apply_accumulated_gradient).

    Layouts, shared with ConvRustArrayLayer (whose im2col is flattened to (N*P, C*k*k)):

    - activations are flat and channel-major, (N, C*H*W), index c*H*W + r*W + col: ConvLayer's
      .nodes order, so a dense layer after the front end has the same weights in both;
    - W[c] is output channel c's kernel in (channel, kernel row, kernel col) order, as
      ConvKernel.weights;
    - im2col is (N, P, C*k*k), P = out_height*out_width in row-major order, columns in W's order.

    The single-example methods are N = 1 wrappers over the batch ones.
    """

    # the constructor keyword arguments after the shape arguments that a subclass takes (as
    # ArrayLayer.hyperparameters); ArrayNetworkBase._new_layer passes them from the network
    hyperparameters: ClassVar[tuple[str, ...]] = ()

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

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.channel_count = channel_count
        self.stride = stride

        self.out_height = (input_height - kernel_size) // stride + 1
        self.out_width = (input_width - kernel_size) // stride + 1
        self.positions = self.out_height * self.out_width

        self.fan_in = input_channels * kernel_size * kernel_size
        self.input_size = input_channels * input_height * input_width
        self.size = channel_count * self.positions

        self.W: FloatArray = np.zeros((channel_count, self.fan_in))
        self.b: FloatArray = np.zeros(channel_count)

        self._grad_W: FloatArray = np.zeros((channel_count, self.fan_in))
        self._grad_b: FloatArray = np.zeros(channel_count)

    def _im2col(self, X: FloatArray) -> FloatArray:
        n = X.shape[0]
        k, s = self.kernel_size, self.stride
        planes = X.reshape(n, self.input_channels, self.input_height, self.input_width)
        # (N, C, H-k+1, W-k+1, k, k) view, strided down to (N, C, out_height, out_width, k, k)
        # numpy 2.2's stub types axis as one int; the function takes a tuple (numpy's docs)
        windows = cast(FloatArray, sliding_window_view(planes, (k, k), axis=(2, 3)))[:, :, ::s, ::s]  # pyright: ignore[reportCallIssue, reportArgumentType]
        # the reshape of the transposed view is the one materializing copy
        return windows.transpose(0, 2, 3, 1, 4, 5).reshape(n, self.positions, self.fan_in)

    def forward_batch(self, X: FloatArray) -> FloatArray:
        n = X.shape[0]
        self._cols = self._im2col(X)
        Z = self._cols @ self.W.T + self.b  # (N, P, channel_count)
        self.Z = Z.transpose(0, 2, 1).reshape(n, self.size)  # channel-major
        self.A = np.maximum(0.0, self.Z)
        return self.A

    def forward(self, x: FloatArray) -> FloatArray:
        self.a = self.forward_batch(x[np.newaxis, :])[0]
        self.z = self.Z[0]
        return self.a

    def compute_output_delta(self, reference: FloatArray) -> None:
        raise NotImplementedError(
            "ConvArrayLayer is a hidden layer, not an output one - see ConvUnit's identical "
            "guard: an unbounded ReLU activation isn't suited to any of this codebase's "
            "output-layer contracts."
        )

    def compute_output_delta_batch(self, reference_batch: FloatArray) -> None:
        self.compute_output_delta(reference_batch)

    def compute_hidden_delta_batch(self, next_layer: ArrayNetworkLayer[FloatArray]) -> None:
        # relu_delta's own convention: derivative 1 where the activation is > 0, 0 otherwise
        # (including exactly z == 0), read off the cached activation
        self.delta_batch = next_layer.downstream_batch() * (self.A > 0.0)

    def compute_hidden_delta(self, next_layer: ArrayNetworkLayer[FloatArray]) -> None:
        self.delta = next_layer.downstream() * (self.a > 0.0)

    def _downstream(self, delta_batch: FloatArray) -> FloatArray:
        # col2im - the vectorized form of ConvLayer._fan_out/downstream_sum
        n = delta_batch.shape[0]
        k, s = self.kernel_size, self.stride
        D = delta_batch.reshape(n, self.channel_count, self.positions)
        dcols = D.transpose(0, 2, 1) @ self.W  # (N, P, C*k*k)
        # -> (N, C, k, k, out_height, out_width), so dcols[:, :, kr, kc] lines up with the input
        # positions kernel offset (kr, kc) reads
        dcols = dcols.reshape(n, self.out_height, self.out_width, self.input_channels, k, k).transpose(0, 3, 4, 5, 1, 2)
        dX = np.zeros((n, self.input_channels, self.input_height, self.input_width))
        row_span = s * (self.out_height - 1) + 1
        col_span = s * (self.out_width - 1) + 1
        # one strided slice-add per kernel offset: within one offset every output position reads
        # a distinct input, so += never collides; overlapping receptive fields across offsets
        # accumulate exactly (no np.add.at needed)
        for kr in range(k):
            for kc in range(k):
                dX[:, :, kr : kr + row_span : s, kc : kc + col_span : s] += dcols[:, :, kr, kc]
        return dX.reshape(n, self.input_size)

    def downstream_batch(self) -> FloatArray:
        return self._downstream(self.delta_batch)

    def downstream(self) -> FloatArray:
        return self._downstream(self.delta[np.newaxis, :])[0]

    def _accumulate(self, delta_batch: FloatArray) -> None:
        # reads the im2col columns forward cached. Positions and batch rows are both summed;
        # apply_accumulated_gradient averages by batch_size only, as ConvKernel does
        n = delta_batch.shape[0]
        D = delta_batch.reshape(n, self.channel_count, self.positions)
        self._grad_W += np.einsum("nop,npk->ok", D, self._cols)
        self._grad_b += D.sum(axis=(0, 2))

    def accumulate_gradient_batch(self, _input_activation_batch: FloatArray) -> None:
        self._accumulate(self.delta_batch)

    def accumulate_gradient(self, _input_activation: FloatArray) -> None:
        self._accumulate(self.delta[np.newaxis, :])

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.W -= learning_rate * (self._grad_W / batch_size)
        self.b -= learning_rate * (self._grad_b / batch_size)
        self._reset_gradient_accum()

    def sgd_step(self, input_activation: FloatArray, learning_rate: float) -> None:
        unfused_sgd_step(self, input_activation, learning_rate)

    def _reset_gradient_accum(self) -> None:
        self._grad_W = np.zeros((self.channel_count, self.fan_in))
        self._grad_b = np.zeros(self.channel_count)
