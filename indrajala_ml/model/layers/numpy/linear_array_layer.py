# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, Z, which strict mode takes for constants)
"""
The linear layer before a batch-norm layer (the batch-norm workplan, D1 and D2): Z = X @ W.T, with
no bias, since the norm layer's mean subtraction cancels one and its beta takes the bias's role
(Ioffe & Szegedy 2015, § 3.2), and no activation, which the norm layer carries.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.specs.hidden_layers import BatchOnly, DeltaIsDownstream, Hidden
from indrajala_ml.model.specs.single_example import refuse_single_example


class LinearArrayLayer(Hidden[FloatArray], BatchOnly[FloatArray], DeltaIsDownstream[FloatArray]):
    """
    Z = X @ W.T, the identity activation, and W stepped by the optimizer (with weight decay). Its
    delta is its downstream from the norm layer after it: the identity's derivative is 1. Hidden
    only, and batch only in training: a network with batch norm has no single-example step.
    """

    decayed: ClassVar[tuple[bool, ...]] = (True,)

    def __init__(self, size: int, input_size: int) -> None:
        self.size = size
        self.input_size = input_size

        self.W: FloatArray = np.zeros((size, input_size))
        self.grad_W: FloatArray = np.zeros((size, input_size))

    def parameters(self) -> tuple[FloatArray, ...]:
        return (self.W,)

    def gradients(self) -> tuple[FloatArray, ...]:
        return (self.grad_W,)

    def set_parameters(self, parameters: Sequence[FloatArray]) -> None:
        (self.W,) = parameters

    def forward(self, x: FloatArray) -> FloatArray:
        # classify_state's single-example forward pass
        self.a = self.W @ x
        return self.a

    def forward_batch(self, X: FloatArray) -> FloatArray:
        self.A = X @ self.W.T
        return self.A

    def downstream_batch(self) -> FloatArray:
        # ArrayLayer.downstream_batch: the gradient into a linear layer before this one
        return self.delta_batch @ self.W

    def accumulate_gradient(self, input_activation: FloatArray) -> None:
        refuse_single_example(self)

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        # ArrayLayer.accumulate_gradient_batch's grad_W
        self.grad_W += self.delta_batch.T @ input_activation_batch

    def reset_gradient_accum(self) -> None:
        self.grad_W = np.zeros((self.size, self.input_size))
