from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from indrajala_ml.model.conv_kernel import ConvKernel
from indrajala_ml.model.conv_unit import ConvUnit
from indrajala_ml.model.protocols.layer_protocols import InputLayer, TrainableLayer
from indrajala_ml.model.specs.window_geometry import output_size, validate_conv_arguments
from indrajala_ml.pcg64 import Pcg64Generator


@dataclass(frozen=True)
class ConvSpec:
    """
    One ConvLayer's hyperparameters; its input shape comes from the previous layer. A conv layer is
    ReLU; a linear one has no bias and no activation, and a BatchNorm follows it (the batch-norm
    workplan, D1 and D2).
    """

    kernel_size: int
    channel_count: int
    stride: int = 1
    activation: Literal["relu", "linear"] = "relu"


class ConvLayer:
    """
    A convolutional hidden layer: channel_count ConvKernels, each shared by every output position of
    its channel and wired to kernel_size x kernel_size receptive fields. Not a BackpropLayer, but it
    implements the layer methods BackpropNetworkBase calls (forward, .nodes, the gradient methods,
    snapshot_state/restore_state, set_training_mode).

    input_layer is a StateLayer or the previous conv or pool layer, read as input_channels
    channel-major planes of input_height x input_width (index c*H*W + r*W + col, the order of this
    layer's own .nodes, so conv layers chain). A receptive field spans every input channel, in the
    kernel weights' (channel, row, col) order. 'valid' padding only.

    Backprop through this layer uses a reverse map built at construction, from each input node to
    the (unit, weight index) pairs that read it: downstream_sum(i) sums over that list only, the
    index form of a full convolution with a flipped kernel, instead of scanning every unit.

    .nodes is channel-major: every (row, col) position for kernel 0, then kernel 1, and so on.
    """

    # the kernel and unit classes; LinearConvLayer's have no bias and no activation
    _kernel_cls: ClassVar[type[ConvKernel]] = ConvKernel
    _unit_cls: ClassVar[type[ConvUnit]] = ConvUnit

    def __init__(
        self,
        input_layer: InputLayer,
        input_height: int,
        input_width: int,
        kernel_size: int,
        channel_count: int,
        stride: int = 1,
        input_channels: int = 1,
    ) -> None:

        validate_conv_arguments(input_height, input_width, input_channels, kernel_size, channel_count, stride)
        assert input_channels * input_height * input_width == len(input_layer.nodes), (
            f"input_channels*input_height*input_width "
            f"({input_channels * input_height * input_width}) must match input_layer's own node "
            f"count ({len(input_layer.nodes)})"
        )

        self.input_layer = input_layer
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.channel_count = channel_count
        self.stride = stride

        self.out_height = output_size(input_height, kernel_size, stride)
        self.out_width = output_size(input_width, kernel_size, stride)

        self.kernels: list[ConvKernel] = [
            self._kernel_cls(kernel_size=kernel_size, in_channels=input_channels) for _ in range(channel_count)
        ]

        self.nodes: list[ConvUnit] = []
        self._fan_out: list[list[tuple[ConvUnit, int]]] = [[] for _ in input_layer.nodes]
        for kernel in self.kernels:
            for row in range(self.out_height):
                for col in range(self.out_width):
                    indices = self._receptive_field_indices(row, col)
                    unit = self._unit_cls(input_nodes=[input_layer.nodes[i] for i in indices], kernel=kernel)
                    self.nodes.append(unit)
                    for weight_index, input_index in enumerate(indices):
                        self._fan_out[input_index].append((unit, weight_index))

    def _receptive_field_indices(self, row: int, col: int) -> list[int]:
        # channel-major, then row-major, into input_layer.nodes: row-major as the loaders decode
        # pixels, channel-major as .nodes
        plane = self.input_height * self.input_width
        return [
            channel * plane + (row * self.stride + kr) * self.input_width + (col * self.stride + kc)
            for channel in range(self.input_channels)
            for kr in range(self.kernel_size)
            for kc in range(self.kernel_size)
        ]

    def forward(self) -> None:
        for unit in self.nodes:
            unit.forward()

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
        # the next layer supplies each unit's downstream sum, dense or sparse
        for own_index, unit in enumerate(self.nodes):
            unit.compute_hidden_delta(next_layer.downstream_sum(own_index))

    def downstream_sum(self, own_index: int) -> float:
        return sum(unit.delta * unit.kernel.weights[weight_index] for unit, weight_index in self._fan_out[own_index])

    def set_training_mode(self, training: bool) -> None:
        # a no-op; BackpropNetworkBase calls it on every trainable layer
        pass

    def accumulate_gradients(self) -> None:
        for unit in self.nodes:
            unit.accumulate_gradient()

    def weight_sets(self) -> Sequence[ConvKernel]:
        return self.kernels

    def snapshot_state(self) -> list[tuple[list[float], float]]:
        return [(list(kernel.weights), kernel.bias) for kernel in self.kernels]

    def restore_state(self, layer_snapshot: list[tuple[list[float], float]]) -> None:
        for kernel, (weights, bias) in zip(self.kernels, layer_snapshot):
            kernel.weights = list(weights)
            kernel.bias = bias

    def randomize_fan_in_aware(self, rng: Pcg64Generator) -> None:
        for kernel in self.kernels:
            kernel.randomize_fan_in_aware(rng)
