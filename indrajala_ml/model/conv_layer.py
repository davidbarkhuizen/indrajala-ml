from __future__ import annotations

from dataclasses import dataclass

from indrajala_ml.model.conv_kernel import ConvKernel
from indrajala_ml.model.conv_unit import ConvUnit
from indrajala_ml.model.state_layer import StateLayer


@dataclass(frozen=True)
class ConvSpec:
    """One ConvLayer's own hyperparameters - everything except its input shape, which a network
    derives from the previous layer (see ConvMultiClassBackpropClassifierNetwork)."""

    kernel_size: int
    channel_count: int
    stride: int = 1


class ConvLayer:
    """
    A convolutional hidden layer - channel_count ConvKernels, each shared across every output
    spatial position in its channel, wired to local kernel_size x kernel_size receptive fields
    of input_layer rather than the whole thing. Not a BackpropLayer subclass (composition, not
    inheritance), but implements the same duck-typed surface
    BackpropNetworkBase's generic machinery relies on (forward/.nodes/accumulate_gradients/
    apply_accumulated_gradients/apply_gradients/snapshot_state/restore_state/set_training_mode -
    see backprop_layer.py's own identical methods).

    input_layer is either a StateLayer or another ConvLayer, read as input_channels
    channel-major planes of input_height x input_width (flat index c*H*W + r*W + col, the same
    ordering this layer's own .nodes uses, so a ConvLayer's output feeds the next ConvLayer
    directly with input_channels=its channel_count). Each receptive field spans every input
    channel, input_channels x kernel_size x kernel_size, and each kernel's flat weights follow
    the same (channel, kernel row, kernel col) order. 'valid' padding only (no synthetic
    zero-padding - output shrinks by kernel_size-1 per stride-1 step).

    Backprop *through* this layer (to a preceding ConvLayer) uses a reverse map built once at
    construction - input node index -> every (unit, kernel weight index) pair whose receptive
    field reads that input node. downstream_sum(i) sums unit.delta * kernel weight over that
    list only: the index form of a "full convolution with a flipped kernel", and sparse, since a
    dense scan over every unit for every input node is O(units * inputs) per example.

    .nodes is channel-major: every (row, col) position for kernel 0, then kernel 1, and so on -
    a stable, documented ordering both this layer's own construction and any downstream dense
    layer's flattened view depend on.
    """

    def __init__(
        self,
        input_layer: StateLayer | ConvLayer,
        input_height: int,
        input_width: int,
        kernel_size: int,
        channel_count: int,
        stride: int = 1,
        input_channels: int = 1,
    ) -> None:

        assert input_channels >= 1, f"input_channels must be at least 1; got {input_channels}"
        assert input_channels * input_height * input_width == len(input_layer.nodes), (
            f"input_channels*input_height*input_width "
            f"({input_channels * input_height * input_width}) must match input_layer's own node "
            f"count ({len(input_layer.nodes)})"
        )
        assert kernel_size >= 1, f"kernel_size must be at least 1; got {kernel_size}"
        assert channel_count >= 1, f"channel_count must be at least 1; got {channel_count}"
        assert stride >= 1, f"stride must be at least 1; got {stride}"
        assert kernel_size <= input_height and kernel_size <= input_width, (
            f"kernel_size ({kernel_size}) must fit within input_height x input_width "
            f"({input_height}x{input_width})"
        )

        self.input_layer = input_layer
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.channel_count = channel_count
        self.stride = stride

        self.out_height = (input_height - kernel_size) // stride + 1
        self.out_width = (input_width - kernel_size) // stride + 1

        self.kernels: list[ConvKernel] = [
            ConvKernel(kernel_size=kernel_size, in_channels=input_channels) for _ in range(channel_count)
        ]

        self.nodes: list[ConvUnit] = []
        self._fan_out: list[list[tuple[ConvUnit, int]]] = [[] for _ in input_layer.nodes]
        for kernel in self.kernels:
            for row in range(self.out_height):
                for col in range(self.out_width):
                    indices = self._receptive_field_indices(row, col)
                    unit = ConvUnit(input_nodes=[input_layer.nodes[i] for i in indices], kernel=kernel)
                    self.nodes.append(unit)
                    for weight_index, input_index in enumerate(indices):
                        self._fan_out[input_index].append((unit, weight_index))

    def _receptive_field_indices(self, row: int, col: int) -> list[int]:
        # channel-major, then row-major flat indexing into input_layer.nodes - row-major
        # matches how mnist_data.py/digits_data.py decode pixels, channel-major matches this
        # layer's own .nodes ordering (see this module's own hot-pixel tests)
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

    def compute_hidden_deltas(self, next_layer) -> None:
        # the next layer (dense BackpropLayer or another ConvLayer) supplies each unit's
        # downstream sum itself, in whichever form (dense or sparse) its own wiring needs
        for own_index, unit in enumerate(self.nodes):
            unit.compute_hidden_delta(next_layer.downstream_sum(own_index))

    def downstream_sum(self, own_index: int) -> float:
        return sum(unit.delta * unit.kernel.weights[weight_index] for unit, weight_index in self._fan_out[own_index])

    def set_training_mode(self, training: bool) -> None:
        # a no-op - see BackpropLayer's own identical no-op default for why every layer needs
        # this method (BackpropNetworkBase._set_training_mode calls it unconditionally on
        # every trainable_layer); DropoutLayer is the one sibling that isn't a no-op
        pass

    def accumulate_gradients(self) -> None:
        for unit in self.nodes:
            unit.accumulate_gradient()

    def apply_accumulated_gradients(self, learning_rate: float, batch_size: int) -> None:
        for kernel in self.kernels:
            kernel.apply_accumulated_gradient(learning_rate, batch_size)

    def apply_gradients(self, learning_rate: float) -> None:
        self.accumulate_gradients()
        self.apply_accumulated_gradients(learning_rate, batch_size=1)

    def snapshot_state(self) -> list[tuple[list[float], float]]:
        return [(list(kernel.weights), kernel.bias) for kernel in self.kernels]

    def restore_state(self, layer_snapshot: list[tuple[list[float], float]]) -> None:
        for kernel, (weights, bias) in zip(self.kernels, layer_snapshot):
            kernel.weights = list(weights)
            kernel.bias = bias

    def randomize_fan_in_aware(self) -> None:
        for kernel in self.kernels:
            kernel.randomize_fan_in_aware()
