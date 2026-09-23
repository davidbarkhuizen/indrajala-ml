from __future__ import annotations

from indrajala_ml.model.conv_kernel import ConvKernel
from indrajala_ml.model.conv_unit import ConvUnit
from indrajala_ml.model.state_layer import StateLayer


class ConvLayer:
    """
    A convolutional hidden layer - channel_count ConvKernels, each shared across every output
    spatial position in its channel, wired to local kernel_size x kernel_size receptive fields
    of input_layer rather than the whole thing. Not a BackpropLayer subclass (composition, not
    inheritance), but implements the same duck-typed surface
    BackpropNetworkBase's generic machinery relies on (forward/.nodes/accumulate_gradients/
    apply_accumulated_gradients/apply_gradients/snapshot_state/restore_state/set_training_mode -
    see backprop_layer.py's own identical methods).

    v1 scope only: input_layer must be a
    plain StateLayer, not another ConvLayer - single input channel, no stacking, since stacking
    needs backprop-through-convolution this layer doesn't implement. 'valid' padding only (no
    synthetic zero-padding - output shrinks by kernel_size-1 per stride-1 step).

    .nodes is channel-major: every (row, col) position for kernel 0, then kernel 1, and so on -
    a stable, documented ordering both this layer's own construction and any downstream dense
    layer's flattened view depend on.
    """

    def __init__(
        self,
        input_layer: StateLayer,
        input_height: int,
        input_width: int,
        kernel_size: int,
        channel_count: int,
        stride: int = 1,
    ) -> None:

        assert input_height * input_width == len(input_layer.nodes), (
            f"input_height*input_width ({input_height * input_width}) must match input_layer's "
            f"own node count ({len(input_layer.nodes)})"
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
        self.kernel_size = kernel_size
        self.channel_count = channel_count
        self.stride = stride

        self.out_height = (input_height - kernel_size) // stride + 1
        self.out_width = (input_width - kernel_size) // stride + 1

        self.kernels: list[ConvKernel] = [
            ConvKernel(kernel_size=kernel_size, in_channels=1) for _ in range(channel_count)
        ]

        self.nodes: list[ConvUnit] = [
            ConvUnit(input_nodes=self._receptive_field_nodes(row, col), kernel=kernel)
            for kernel in self.kernels
            for row in range(self.out_height)
            for col in range(self.out_width)
        ]

    def _receptive_field_nodes(self, row: int, col: int) -> list:
        # row-major flat indexing into input_layer.nodes - matches how mnist_data.py/
        # digits_data.py decode pixels (see this module's own hot-pixel test)
        indices = [
            (row * self.stride + kr) * self.input_width + (col * self.stride + kc)
            for kr in range(self.kernel_size)
            for kc in range(self.kernel_size)
        ]
        return [self.input_layer.nodes[i] for i in indices]

    def forward(self) -> None:
        for unit in self.nodes:
            unit.forward()

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
