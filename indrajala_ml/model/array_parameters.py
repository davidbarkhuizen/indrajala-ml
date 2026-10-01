# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, which strict mode takes for constants)
"""
The optimizer's accessors (array_protocols.TrainedArrayLayer) for a layer whose parameters are W and
b, every dense and conv layer, and for a batch-norm layer's gamma and beta, on both backends.
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


class GammaAndBeta[A]:
    """
    A batch-norm layer's (gamma, beta), neither decayed (the batch-norm workplan, D7), and its
    running averages (running_mean, running_var), which the network saves but no optimizer trains.
    """

    decayed: ClassVar[tuple[bool, ...]] = (False, False)

    gamma: A
    beta: A
    grad_gamma: A
    grad_beta: A
    running_mean: A
    running_var: A

    def parameters(self) -> tuple[A, ...]:
        return self.gamma, self.beta

    def gradients(self) -> tuple[A, ...]:
        return self.grad_gamma, self.grad_beta

    def set_parameters(self, parameters: Sequence[A]) -> None:
        self.gamma, self.beta = parameters

    def running_state(self) -> tuple[A, ...]:
        return self.running_mean, self.running_var

    def set_running_state(self, state: Sequence[A]) -> None:
        self.running_mean, self.running_var = state
