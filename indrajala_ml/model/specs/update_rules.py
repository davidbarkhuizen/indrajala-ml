"""
The update rules an optimizer applies (optimizers/), as frozen dataclasses holding only their
hyperparameters (the composable-layers workplan, D2). Each follows a published form with the
source's arithmetic grouping (README, Update rules). g is the gradient summed over a batch of B
examples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Kingma & Ba (2014)'s published defaults. Unlike momentum's coefficient these have defaults:
# in practice they are near-fixed constants, not a tuned knob.
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True)
class SGD:
    """w - lr * (g / B): Goyal et al. 2017, eq. (2)."""


@dataclass(frozen=True)
class Momentum:
    """u = m * u + g / B; w - lr * u: Goyal et al. 2017, eq. (9). momentum is required."""

    momentum: float


@dataclass(frozen=True)
class Adam:
    """
    Kingma & Ba 2014, Algorithm 1, on g / B, with one step count t for the whole network; with
    weight_decay, AdamW (Loshchilov & Hutter 2019, Algorithm 2; the training-recipe workplan, D9):
    decoupled weight decay on the weights only (biases, gamma, beta and the position and embedding
    tables aren't decayed, as WeightDecay treats them), in PyTorch's AdamW grouping, w * (1 - lr *
    weight_decay) - lr * m_hat / (sqrt(v_hat) + epsilon). At 0, the default, the factor is exactly
    1.0: Adam's bits.
    """

    beta1: float = DEFAULT_BETA1
    beta2: float = DEFAULT_BETA2
    epsilon: float = DEFAULT_EPSILON
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.weight_decay) and self.weight_decay >= 0.0):
            raise ValueError(f"weight_decay must be finite and at least 0; got {self.weight_decay!r}")

    def decay_factor(self, learning_rate: float) -> float:
        """1 - learning_rate * weight_decay, a decayed weight's factor before the Adam step."""
        return 1.0 - learning_rate * self.weight_decay

    def bias_corrections(self, t: int) -> tuple[float, float]:
        """1 - beta1^t and 1 - beta2^t, every implementation's (the crate's in src/fused.rs)."""
        return 1.0 - powi(self.beta1, t), 1.0 - powi(self.beta2, t)


def powi(base: float, exponent: int) -> float:
    """
    base^exponent for exponent >= 0 by square-and-multiply from the exponent's low bit, the crate's
    order (src/scalar.rs; the aarch64 workplan's D6, retired: git show 6b9c6fe:docs/workplans/infrastructure/aarch64-workplan.md), so the three implementations agree; the C
    library's pow, which ** calls, rounds differently.
    """
    result = 1.0
    while True:
        if exponent & 1:
            result *= base
        exponent >>= 1
        if exponent == 0:
            return result
        base *= base


@dataclass(frozen=True)
class WeightDecay:
    """w - lr * (g / B + l2_lambda * w), the bias plain SGD: Goyal et al. 2017, eq. (8)."""

    l2_lambda: float


UpdateRule = SGD | Momentum | Adam | WeightDecay
