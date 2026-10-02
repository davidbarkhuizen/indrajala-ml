# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, A, which strict mode takes for constants)
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.array_parameters import WeightAndBias
from indrajala_ml.model.protocols.array_protocols import ArrayNetworkLayer
from indrajala_ml.model.specs.single_example import refuse_single_example
from indrajala_ml.model.specs.window_geometry import output_size, validate_conv_arguments


class ConvGeometryArrayLayer:
    """
    What ConvArrayLayer and LinearConvArrayLayer share: the geometry, the kernels W, im2col for
    the forward pass, col2im for the downstream, and grad_W.
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

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.channel_count = channel_count
        self.stride = stride

        self.out_height = output_size(input_height, kernel_size, stride)
        self.out_width = output_size(input_width, kernel_size, stride)
        self.positions = self.out_height * self.out_width

        self.fan_in = input_channels * kernel_size * kernel_size
        self.input_size = input_channels * input_height * input_width
        self.size = channel_count * self.positions

        self.W: FloatArray = np.zeros((channel_count, self.fan_in))
        self.grad_W: FloatArray = np.zeros((channel_count, self.fan_in))

    def _im2col(self, X: FloatArray) -> FloatArray:
        n = X.shape[0]
        k, s = self.kernel_size, self.stride
        planes = X.reshape(n, self.input_channels, self.input_height, self.input_width)
        # (N, C, H-k+1, W-k+1, k, k) view, strided down to (N, C, out_height, out_width, k, k)
        windows = sliding_window_view(planes, (k, k), axis=(2, 3))[:, :, ::s, ::s]
        # the reshape of the transposed view is the one materializing copy
        return windows.transpose(0, 2, 3, 1, 4, 5).reshape(n, self.positions, self.fan_in)

    def _products(self, X: FloatArray) -> FloatArray:
        # the kernels' products, (N, P, channel_count), caching the im2col columns
        self._cols = self._im2col(X)
        return self._cols @ self.W.T

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

    def _accumulate_W(self, D: FloatArray) -> None:
        # D is the delta as (N, channel_count, P); reads the im2col columns forward cached
        self.grad_W += np.einsum("nop,npk->ok", D, self._cols)


class ConvArrayLayer(WeightAndBias[FloatArray], ConvGeometryArrayLayer):
    """
    ConvLayer (conv_layer.py) over numpy arrays: a ReLU convolutional hidden layer, 'valid'
    padding, with the kernels as one matrix and a batch's receptive fields as one im2col array.

    Not an ArrayLayer: size is the flattened output count (channel_count * out_height * out_width),
    while W is (channel_count, input_channels * kernel_size**2). It implements the methods
    ArrayNetworkBase calls (forward*, compute_*_delta*, downstream*, accumulate_gradient*), and
    the network's optimizer (optimizers/) steps W and b as a dense layer's.

    Layouts, shared with ConvRustArrayLayer (whose im2col is flattened to (N*P, C*k*k)):

    - activations are flat and channel-major, (N, C*H*W), index c*H*W + r*W + col: ConvLayer's
      .nodes order, so a dense layer after the front end has the same weights in both;
    - W[c] is output channel c's kernel in (channel, kernel row, kernel col) order, as
      ConvKernel.weights;
    - im2col is (N, P, C*k*k), P = out_height*out_width in row-major order, columns in W's order.

    The single-example methods are N = 1 wrappers over the batch ones.
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
        super().__init__(input_height, input_width, input_channels, kernel_size, channel_count, stride)
        self.b: FloatArray = np.zeros(channel_count)
        self.grad_b: FloatArray = np.zeros(channel_count)

    def forward_batch(self, X: FloatArray) -> FloatArray:
        n = X.shape[0]
        Z = self._products(X) + self.b  # (N, P, channel_count)
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

    def downstream_batch(self) -> FloatArray:
        return self._downstream(self.delta_batch)

    def downstream(self) -> FloatArray:
        return self._downstream(self.delta[np.newaxis, :])[0]

    def _accumulate(self, delta_batch: FloatArray) -> None:
        # reads the im2col columns forward cached. Positions and batch rows are both summed; the
        # optimizer averages by batch_size only, as the pure-Python optimizer does for ConvKernel
        n = delta_batch.shape[0]
        D = delta_batch.reshape(n, self.channel_count, self.positions)
        self._accumulate_W(D)
        self.grad_b += D.sum(axis=(0, 2))

    def accumulate_gradient_batch(self, _input_activation_batch: FloatArray) -> None:
        self._accumulate(self.delta_batch)

    def accumulate_gradient(self, _input_activation: FloatArray) -> None:
        self._accumulate(self.delta[np.newaxis, :])

    def reset_gradient_accum(self) -> None:
        self.grad_W = np.zeros((self.channel_count, self.fan_in))
        self.grad_b = np.zeros(self.channel_count)


class LinearConvArrayLayer(ConvGeometryArrayLayer):
    """
    The conv layer before a batch-norm layer (the batch-norm workplan, D1 and D2): ConvArrayLayer's
    products, without the bias and the ReLU, in the same channel-major layout. Its delta is its
    downstream from the norm layer after it, and W is stepped by the optimizer (with weight decay).
    Hidden only, and batch only in training: a network with batch norm has no single-example step.
    """

    decayed: ClassVar[tuple[bool, ...]] = (True,)

    def parameters(self) -> tuple[FloatArray, ...]:
        return (self.W,)

    def gradients(self) -> tuple[FloatArray, ...]:
        return (self.grad_W,)

    def set_parameters(self, parameters: Sequence[FloatArray]) -> None:
        (self.W,) = parameters

    def forward_batch(self, X: FloatArray) -> FloatArray:
        n = X.shape[0]
        self.A = self._products(X).transpose(0, 2, 1).reshape(n, self.size)  # channel-major
        return self.A

    def forward(self, x: FloatArray) -> FloatArray:
        # classify_state's single-example forward pass
        self.a = self.forward_batch(x[np.newaxis, :])[0]
        return self.a

    def compute_output_delta(self, reference: FloatArray) -> None:
        raise NotImplementedError("a linear conv layer is hidden, before a batch-norm layer")

    def compute_output_delta_batch(self, reference_batch: FloatArray) -> None:
        raise NotImplementedError("a linear conv layer is hidden, before a batch-norm layer")

    def compute_hidden_delta(self, next_layer: Any) -> None:
        refuse_single_example(self)

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        self.delta_batch = next_layer.downstream_batch()

    def downstream(self) -> FloatArray:
        refuse_single_example(self)

    def downstream_batch(self) -> FloatArray:
        return self._downstream(self.delta_batch)

    def accumulate_gradient(self, input_activation: FloatArray) -> None:
        refuse_single_example(self)

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        n = self.delta_batch.shape[0]
        self._accumulate_W(self.delta_batch.reshape(n, self.channel_count, self.positions))

    def reset_gradient_accum(self) -> None:
        self.grad_W = np.zeros((self.channel_count, self.fan_in))
