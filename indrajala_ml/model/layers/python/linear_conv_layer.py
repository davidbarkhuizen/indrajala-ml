"""
The pure-Python conv layer before a batch-norm layer (the batch-norm workplan, D1 and D2), the
counterpart of conv_array_layer.LinearConvArrayLayer: each unit's z, ConvUnit's weighted sum
without the kernel's bias, and no activation. The norm layer's mean subtraction cancels a bias, and
its beta takes the bias's role (Ioffe & Szegedy 2015, § 3.2). The norm layer carries the activation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from indrajala_ml.model.layers.python.conv_kernel import ConvKernel
from indrajala_ml.model.layers.python.conv_layer import ConvLayer
from indrajala_ml.model.layers.python.conv_unit import ConvUnit
from indrajala_ml.model.layers.python.fan_in_aware_init import fan_in_aware_weights
from indrajala_ml.pcg64 import Pcg64Generator


class LinearConvKernel(ConvKernel):
    """
    A ConvKernel without a bias: its bias stays 0.0, and neither the optimizer (has_bias), a draw
    nor a snapshot touches it.
    """

    has_bias: ClassVar[bool] = False

    def randomize_fan_in_aware(self, rng: Pcg64Generator) -> None:
        # weights only, as LinearLayer draws
        self.weights = fan_in_aware_weights(rng, len(self.weights))

    def accumulate_gradient(self, delta: float, receptive_field_values: Sequence[float]) -> None:
        # ConvKernel's, without the bias
        assert len(receptive_field_values) == len(self.weights)
        for i, value in enumerate(receptive_field_values):
            self.weight_gradient_accum[i] += delta * value


class LinearConvUnit(ConvUnit):
    """A ConvUnit with the identity activation and no bias; its delta is its downstream sum."""

    def forward(self) -> float:
        # ConvUnit.z()'s sum, without the bias
        self._activation = sum(node.value() * weight for node, weight in zip(self.input_nodes, self.kernel.weights))
        return self._activation

    def compute_hidden_delta(self, downstream_sum: float) -> None:
        # the identity's derivative is 1
        self.delta = downstream_sum


class LinearConvLayer(ConvLayer):
    """
    A ConvLayer of LinearConvKernels and LinearConvUnits, trained through the layer-major batch path
    only (layer_major.py): a network with batch norm has no single-example step (D4).
    """

    _kernel_cls = LinearConvKernel
    _unit_cls = LinearConvUnit

    def snapshot_state(self) -> list[tuple[list[float]]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return [(list(kernel.weights),) for kernel in self.kernels]

    def restore_state(self, layer_snapshot: Sequence[Sequence[list[float]]]) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        for kernel, (weights,) in zip(self.kernels, layer_snapshot):
            kernel.weights = list(weights)
