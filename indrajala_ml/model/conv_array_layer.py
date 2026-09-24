from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from indrajala_ml.model.array_layer import unfused_sgd_step


def validate_conv_arguments(
    input_height: int, input_width: int, input_channels: int, kernel_size: int, channel_count: int, stride: int
) -> None:
    # the same assertions as ConvLayer's own constructor (minus the input_layer node-count check -
    # there's no input_layer object here), shared with ConvRustArrayLayer
    assert input_channels >= 1, f"input_channels must be at least 1; got {input_channels}"
    assert kernel_size >= 1, f"kernel_size must be at least 1; got {kernel_size}"
    assert channel_count >= 1, f"channel_count must be at least 1; got {channel_count}"
    assert stride >= 1, f"stride must be at least 1; got {stride}"
    assert kernel_size <= input_height and kernel_size <= input_width, (
        f"kernel_size ({kernel_size}) must fit within input_height x input_width "
        f"({input_height}x{input_width})"
    )


class ConvArrayLayer:
    """
    The numpy counterpart to ConvLayer (conv_layer.py): a ReLU convolutional hidden layer, 'valid'
    padding, with the whole layer's kernels as one matrix and every receptive field of a batch as
    one im2col array, instead of one ConvUnit object per output position.

    Not an ArrayLayer subclass, mirroring ConvLayer not subclassing BackpropLayer: ArrayLayer's
    size/input_size define its W and gradient-accumulator shapes, but here size is the flattened
    output count (channel_count * out_height * out_width - what the next layer chains from) while
    W is (channel_count, input_channels * kernel_size**2). Implements the same duck-typed surface
    ArrayNetworkBase drives (forward*/compute_*_delta*/downstream*/accumulate_gradient*/
    apply_accumulated_gradient).

    Layouts, fixed so a Rust layer can mirror them later:

    - activations are flat at the layer boundary, (N, C*H*W) channel-major (flat index
      c*H*W + r*W + col) - ConvLayer's own .nodes ordering, so a dense layer after the conv front
      end has the same weight matrix in both implementations;
    - W[c] is output channel c's kernel in (channel, kernel row, kernel col) order - exactly
      ConvKernel.weights, so W[c] = kernel.weights injects identical weights;
    - im2col columns are (N, P, C*k*k), P = out_height*out_width in row-major output order,
      column order matching W's rows.

    The single-example path (forward/compute_hidden_delta/downstream/accumulate_gradient) is a
    thin N = 1 wrapper over the batch path, so the maths has one implementation.
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

        self.out_height = (input_height - kernel_size) // stride + 1
        self.out_width = (input_width - kernel_size) // stride + 1
        self.positions = self.out_height * self.out_width

        self.fan_in = input_channels * kernel_size * kernel_size
        self.input_size = input_channels * input_height * input_width
        self.size = channel_count * self.positions

        self.W: np.ndarray = np.zeros((channel_count, self.fan_in))
        self.b: np.ndarray = np.zeros(channel_count)

        self._grad_W: np.ndarray = np.zeros((channel_count, self.fan_in))
        self._grad_b: np.ndarray = np.zeros(channel_count)

    def _im2col(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        k, s = self.kernel_size, self.stride
        planes = X.reshape(n, self.input_channels, self.input_height, self.input_width)
        # (N, C, H-k+1, W-k+1, k, k) view, strided down to (N, C, out_height, out_width, k, k)
        windows = sliding_window_view(planes, (k, k), axis=(2, 3))[:, :, ::s, ::s]
        # the reshape of the transposed view is the one materializing copy
        return windows.transpose(0, 2, 3, 1, 4, 5).reshape(n, self.positions, self.fan_in)

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        self._cols = self._im2col(X)
        Z = self._cols @ self.W.T + self.b  # (N, P, channel_count)
        self.Z = Z.transpose(0, 2, 1).reshape(n, self.size)  # channel-major
        self.A = np.maximum(0.0, self.Z)
        return self.A

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.a = self.forward_batch(x[np.newaxis, :])[0]
        self.z = self.Z[0]
        return self.a

    def compute_output_delta(self, reference: np.ndarray) -> None:
        raise NotImplementedError(
            "ConvArrayLayer is a hidden layer, not an output one - see ConvUnit's identical "
            "guard: an unbounded ReLU activation isn't suited to any of this codebase's "
            "output-layer contracts."
        )

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.compute_output_delta(reference_batch)

    def compute_hidden_delta_batch(self, next_layer) -> None:
        # relu_delta's own convention: derivative 1 where the activation is > 0, 0 otherwise
        # (including exactly z == 0), read off the cached activation
        self.delta_batch = next_layer.downstream_batch() * (self.A > 0.0)

    def compute_hidden_delta(self, next_layer) -> None:
        self.delta = next_layer.downstream() * (self.a > 0.0)

    def _downstream(self, delta_batch: np.ndarray) -> np.ndarray:
        # col2im - the vectorized form of ConvLayer._fan_out/downstream_sum
        n = delta_batch.shape[0]
        k, s = self.kernel_size, self.stride
        D = delta_batch.reshape(n, self.channel_count, self.positions)
        dcols = D.transpose(0, 2, 1) @ self.W  # (N, P, C*k*k)
        # -> (N, C, k, k, out_height, out_width), so dcols[:, :, kr, kc] lines up with the input
        # positions kernel offset (kr, kc) reads
        dcols = dcols.reshape(n, self.out_height, self.out_width, self.input_channels, k, k).transpose(
            0, 3, 4, 5, 1, 2
        )
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

    def downstream_batch(self) -> np.ndarray:
        return self._downstream(self.delta_batch)

    def downstream(self) -> np.ndarray:
        return self._downstream(self.delta[np.newaxis, :])[0]

    def _accumulate(self, delta_batch: np.ndarray) -> None:
        # uses the im2col columns forward cached rather than rebuilding them from the input
        # activation - forward always precedes this in ArrayNetworkBase.learn/learn_batch. Spatial
        # positions and batch rows are both summed here; apply_accumulated_gradient averages by
        # batch_size only, the same composition ConvKernel documents.
        n = delta_batch.shape[0]
        D = delta_batch.reshape(n, self.channel_count, self.positions)
        self._grad_W += np.einsum("nop,npk->ok", D, self._cols)
        self._grad_b += D.sum(axis=(0, 2))

    def accumulate_gradient_batch(self, _input_activation_batch: np.ndarray) -> None:
        self._accumulate(self.delta_batch)

    def accumulate_gradient(self, _input_activation: np.ndarray) -> None:
        self._accumulate(self.delta[np.newaxis, :])

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.W -= learning_rate * self._grad_W / batch_size
        self.b -= learning_rate * self._grad_b / batch_size
        self._reset_gradient_accum()

    def sgd_step(self, input_activation: np.ndarray, learning_rate: float) -> None:
        unfused_sgd_step(self, input_activation, learning_rate)

    def _reset_gradient_accum(self) -> None:
        self._grad_W = np.zeros((self.channel_count, self.fan_in))
        self._grad_b = np.zeros(self.channel_count)
