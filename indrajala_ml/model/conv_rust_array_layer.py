# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, A, which strict mode takes for constants)
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import indrajala_math_rust as pa

from indrajala_ml.model.array_parameters import WeightAndBias
from indrajala_ml.model.array_protocols import ArrayNetworkLayer
from indrajala_ml.model.conv_array_layer import validate_conv_arguments
from indrajala_ml.model.layer_specs import refuse_single_example


class ConvGeometryRustArrayLayer:
    """
    What ConvRustArrayLayer and LinearConvRustArrayLayer share, as ConvGeometryArrayLayer for the
    numpy layers: the geometry (one pa.ConvGeometry, passed to every call), the kernels W, grad_W,
    and the downstream.
    """

    delta_batch: pa.Array

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
        self.grad_W = pa.Array.zeros((channel_count, self.fan_in))

    def downstream_batch(self) -> pa.Array:
        return pa.conv_downstream_batch(self.W, self.delta_batch, self.geometry)


class ConvRustArrayLayer(WeightAndBias[pa.Array], ConvGeometryRustArrayLayer):
    """
    ConvArrayLayer on the Rust backend: the same ReLU conv layer and layouts, each method one fused
    call (conv.rs). Conv tensors cross as matrices, since indrajala_math_rust.Array is 1D/2D only.

    It keeps no pre-activation Z: the forward op applies the ReLU as it writes A. The backward pass
    masks on A (derivative 0 at exactly z == 0), as ConvArrayLayer's does.

    Single-example calls pass 1D arrays straight to the batch ops, which take a vector as N = 1:
    pa.Array.reshape copies, unlike numpy's x[np.newaxis].
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
        self.b = pa.Array.zeros(channel_count)
        self.grad_b = pa.Array.zeros(channel_count)

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A, self._cols = pa.conv_forward_batch(self.W, X, self.b, self.geometry)
        return self.A

    def forward(self, x: pa.Array) -> pa.Array:
        self.a, self._cols = pa.conv_forward_batch(self.W, x, self.b, self.geometry)
        return self.a

    def compute_output_delta(self, reference: pa.Array) -> None:
        raise NotImplementedError(
            "ConvRustArrayLayer is a hidden layer, not an output one - see ConvUnit's identical "
            "guard: an unbounded ReLU activation isn't suited to any of this codebase's "
            "output-layer contracts."
        )

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        self.compute_output_delta(reference_batch)

    def compute_hidden_delta_batch(self, next_layer: ArrayNetworkLayer[pa.Array]) -> None:
        # array_relu_mask: derivative 0 at exactly z == 0, the same convention as ConvArrayLayer
        self.delta_batch = pa.array_relu_mask(next_layer.downstream_batch(), self.A)

    def compute_hidden_delta(self, next_layer: ArrayNetworkLayer[pa.Array]) -> None:
        self.delta = pa.array_relu_mask(next_layer.downstream(), self.a)

    def downstream(self) -> pa.Array:
        return pa.conv_downstream_batch(self.W, self.delta, self.geometry)

    def accumulate_gradient_batch(self, _input_activation_batch: pa.Array) -> None:
        # reads the im2col columns forward cached, as ConvArrayLayer._accumulate does
        self.grad_W, self.grad_b = pa.conv_accumulate_gradient_batch(
            self.delta_batch, self._cols, self.grad_W, self.grad_b, self.geometry
        )

    def accumulate_gradient(self, _input_activation: pa.Array) -> None:
        self.grad_W, self.grad_b = pa.conv_accumulate_gradient_batch(
            self.delta, self._cols, self.grad_W, self.grad_b, self.geometry
        )

    def reset_gradient_accum(self) -> None:
        self.grad_W = pa.Array.zeros((self.channel_count, self.fan_in))
        self.grad_b = pa.Array.zeros(self.channel_count)


class LinearConvRustArrayLayer(ConvGeometryRustArrayLayer):
    """
    LinearConvArrayLayer on the Rust backend: ConvRustArrayLayer's products without the bias and
    the ReLU, before a batch-norm layer, each method one call (conv.rs's conv_linear_*). Its delta
    is its downstream from the norm layer after it. Hidden only, and batch only in training.
    """

    decayed: ClassVar[tuple[bool, ...]] = (True,)

    def parameters(self) -> tuple[pa.Array, ...]:
        return (self.W,)

    def gradients(self) -> tuple[pa.Array, ...]:
        return (self.grad_W,)

    def set_parameters(self, parameters: Sequence[pa.Array]) -> None:
        (self.W,) = parameters

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A, self._cols = pa.conv_linear_forward_batch(self.W, X, self.geometry)
        return self.A

    def forward(self, x: pa.Array) -> pa.Array:
        # classify_state's single-example forward pass
        self.a, self._cols = pa.conv_linear_forward_batch(self.W, x, self.geometry)
        return self.a

    def compute_output_delta(self, reference: pa.Array) -> None:
        raise NotImplementedError("a linear conv layer is hidden, before a batch-norm layer")

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        raise NotImplementedError("a linear conv layer is hidden, before a batch-norm layer")

    def compute_hidden_delta(self, next_layer: Any) -> None:
        refuse_single_example(self)

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        self.delta_batch = next_layer.downstream_batch()

    def downstream(self) -> pa.Array:
        refuse_single_example(self)

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        refuse_single_example(self)

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        # reads the im2col columns forward cached
        self.grad_W = pa.conv_linear_accumulate_gradient_batch(self.delta_batch, self._cols, self.grad_W, self.geometry)

    def reset_gradient_accum(self) -> None:
        self.grad_W = pa.Array.zeros((self.channel_count, self.fan_in))
