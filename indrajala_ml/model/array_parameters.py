# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, which strict mode takes for constants)
"""
The optimizer's accessor (array_protocols.TrainedArrayLayer) for a layer whose parameters are W and
b: every dense and conv layer, on both backends.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar


class WeightAndBias[A]:
    """(W, b), with W decayed under WeightDecay and b not."""

    decayed: ClassVar[tuple[bool, ...]] = (True, False)

    W: A
    b: A
    grad_W: A
    grad_b: A

    def parameters(self) -> tuple[A, ...]:
        return self.W, self.b

    def gradients(self) -> tuple[A, ...]:
        return self.grad_W, self.grad_b

    def set_parameters(self, parameters: Sequence[A]) -> None:
        self.W, self.b = parameters
