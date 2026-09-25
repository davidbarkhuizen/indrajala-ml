from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def validate_pool_arguments(
    input_height: int, input_width: int, input_channels: int, pool_size: int, stride: int
) -> None:
    # MaxPoolLayer's constructor checks, less the input_layer node count; shared with
    # MaxPoolRustArrayLayer
    assert pool_size >= 1, f"pool_size must be at least 1; got {pool_size}"
    assert stride >= 1, f"stride must be at least 1; got {stride}"
    assert input_channels >= 1, f"input_channels must be at least 1; got {input_channels}"
    assert pool_size <= input_height and pool_size <= input_width, (
        f"pool_size ({pool_size}) must fit within input_height x input_width ({input_height}x{input_width})"
    )


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

        stride = pool_size if stride is None else stride
        validate_pool_arguments(input_height, input_width, input_channels, pool_size, stride)

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.pool_size = pool_size
        self.stride = stride
        self.channel_count = input_channels

        self.out_height = (input_height - pool_size) // stride + 1
        self.out_width = (input_width - pool_size) // stride + 1

        self.input_size = input_channels * input_height * input_width
        self.size = input_channels * self.out_height * self.out_width

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        p, s = self.pool_size, self.stride
        planes = X.reshape(n, self.input_channels, self.input_height, self.input_width)
        # numpy 2.2's stub types axis as one int; the function takes a tuple (numpy's docs)
        windows = sliding_window_view(planes, (p, p), axis=(2, 3))[:, :, ::s, ::s]  # pyright: ignore[reportCallIssue, reportArgumentType]
        slots = windows.reshape(n, self.input_channels, self.out_height, self.out_width, p * p)
        self.argmax_batch = slots.argmax(axis=-1)  # (N, C, out_height, out_width)
        self.A = np.take_along_axis(slots, self.argmax_batch[..., np.newaxis], axis=-1).reshape(n, self.size)
        return self.A

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.a = self.forward_batch(x[np.newaxis, :])[0]
        self.argmax = self.argmax_batch[0]
        return self.a

    def compute_output_delta(self, reference: np.ndarray) -> None:
        raise NotImplementedError("MaxPoolArrayLayer is a hidden layer, not an output one.")

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.compute_output_delta(reference_batch)

    def compute_hidden_delta_batch(self, next_layer) -> None:
        # max is the identity on its winning input - no activation derivative to multiply in
        self.delta_batch = next_layer.downstream_batch()

    def compute_hidden_delta(self, next_layer) -> None:
        self.delta = next_layer.downstream()

    def _downstream(self, delta_batch: np.ndarray, argmax_batch: np.ndarray) -> np.ndarray:
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

    def downstream_batch(self) -> np.ndarray:
        return self._downstream(self.delta_batch, self.argmax_batch)

    def downstream(self) -> np.ndarray:
        return self._downstream(self.delta[np.newaxis, :], self.argmax[np.newaxis])[0]

    # weight-free: every gradient hook below is a deliberate no-op

    def accumulate_gradient_batch(self, _input_activation_batch: np.ndarray) -> None:
        pass

    def accumulate_gradient(self, _input_activation: np.ndarray) -> None:
        pass

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        pass

    def sgd_step(self, _input_activation: np.ndarray, learning_rate: float) -> None:
        pass
