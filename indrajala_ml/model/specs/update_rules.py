"""
The update rules an optimizer applies (optimizers/), as frozen dataclasses holding only their
hyperparameters (the composable-layers workplan, D2). Each follows a published form with the
source's arithmetic grouping (README, Update rules). g is the gradient summed over a batch of B
examples.
"""

from __future__ import annotations

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
    """Kingma & Ba 2014, Algorithm 1, on g / B, with one step count t for the whole network."""

    beta1: float = DEFAULT_BETA1
    beta2: float = DEFAULT_BETA2
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True)
class WeightDecay:
    """w - lr * (g / B + l2_lambda * w), the bias plain SGD: Goyal et al. 2017, eq. (8)."""

    l2_lambda: float


UpdateRule = SGD | Momentum | Adam | WeightDecay
