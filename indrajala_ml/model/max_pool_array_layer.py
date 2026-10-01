from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.array_protocols import ArrayNetworkLayer
from indrajala_ml.model.window_geometry import output_size, pool_stride, validate_pool_arguments

# each window's winning slot index, as np.argmax returns it
IndexArray = npt.NDArray[np.intp]


class MaxPoolArrayLayer:
    """
    MaxPoolLayer (max_pool_layer.py) over numpy arrays: each of input_channels channel-major planes
    pooled separately (channel_count == input_channels), with ConvArrayLayer's flat channel-major
    activations.

    Window slots are numbered row-major (pr, pc), and np.argmax takes the first maximal slot, as
    PoolUnit does, so exact ties (all-zero windows after a ReLU) pick the same winner.

    No weights: the gradient methods are no-ops. The single-example methods are N = 1 wrappers over
    the batch ones.
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

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.pool_size = pool_size
        self.stride = stride
        self.channel_count = input_channels

        self.out_height = output_size(input_height, pool_size, stride)
        self.out_width = output_size(input_width, pool_size, stride)

        self.input_size = input_channels * input_height * input_width
        self.size = input_channels * self.out_height * self.out_width

    def forward_batch(self, X: FloatArray) -> FloatArray:
        n = X.shape[0]
        p, s = self.pool_size, self.stride
        planes = X.reshape(n, self.input_channels, self.input_height, self.input_width)
        windows = sliding_window_view(planes, (p, p), axis=(2, 3))[:, :, ::s, ::s]
        slots = windows.reshape(n, self.input_channels, self.out_height, self.out_width, p * p)
        self.argmax_batch: IndexArray = slots.argmax(axis=-1)  # (N, C, out_height, out_width)
        self.A = np.take_along_axis(slots, self.argmax_batch[..., np.newaxis], axis=-1).reshape(n, self.size)
        return self.A

    def forward(self, x: FloatArray) -> FloatArray:
        self.a = self.forward_batch(x[np.newaxis, :])[0]
        self.argmax = self.argmax_batch[0]
        return self.a

    def compute_output_delta(self, reference: FloatArray) -> None:
        raise NotImplementedError("MaxPoolArrayLayer is a hidden layer, not an output one.")

    def compute_output_delta_batch(self, reference_batch: FloatArray) -> None:
        self.compute_output_delta(reference_batch)

    def compute_hidden_delta_batch(self, next_layer: ArrayNetworkLayer[FloatArray]) -> None:
        # max is the identity on its winning input - no activation derivative to multiply in
        self.delta_batch = next_layer.downstream_batch()

    def compute_hidden_delta(self, next_layer: ArrayNetworkLayer[FloatArray]) -> None:
        self.delta = next_layer.downstream()

    def _downstream(self, delta_batch: FloatArray, argmax_batch: IndexArray) -> FloatArray:
        # the vectorized form of MaxPoolLayer.downstream_sum: each window's delta goes to its
        # winning slot only. One masked strided slice-add per slot (pr, pc) - within one slot
        # every window reads a distinct input, so += never collides; with overlapping windows
        # (stride < pool_size) an input that won several windows receives every one of their
        # deltas across slots.
        n = delta_batch.shape[0]
        p, s = self.pool_size, self.stride
        D = delta_batch.reshape(n, self.input_channels, self.out_height, self.out_width)
        dX = np.zeros((n, self.input_channels, self.input_height, self.input_width))
        row_span = s * (self.out_height - 1) + 1
        col_span = s * (self.out_width - 1) + 1
        for slot in range(p * p):
            pr, pc = divmod(slot, p)
            dX[:, :, pr : pr + row_span : s, pc : pc + col_span : s] += D * (argmax_batch == slot)
        return dX.reshape(n, self.input_size)

    def downstream_batch(self) -> FloatArray:
        return self._downstream(self.delta_batch, self.argmax_batch)

    def downstream(self) -> FloatArray:
        return self._downstream(self.delta[np.newaxis, :], self.argmax[np.newaxis])[0]

    # weight-free: every gradient hook below is a deliberate no-op, and with no W the optimizer
    # skips the layer

    def accumulate_gradient_batch(self, _input_activation_batch: FloatArray) -> None:
        pass

    def accumulate_gradient(self, _input_activation: FloatArray) -> None:
        pass
