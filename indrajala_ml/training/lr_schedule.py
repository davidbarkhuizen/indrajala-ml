"""
Learning-rate schedules as data (the training-recipe workplan, D2-D4): frozen dataclasses, each a
callable from a step (the trainer's iteration count, batches) to a rate, so a run file can record
and check one (D5), as the update rules are recorded (README, Update rules).

- Constant(rate): rate at every step.
- Warmup(steps, then): linear from then(0) / steps at step 0 to then(0) at step steps - 1, then
  `then`, counted from the end of warmup. Warmup(steps, Constant(rate)) is the warmup Goyal et al.
  2017 pair with the linear scaling rule (a scaled rate from step one is unstable: batch 128's rate
  diverged to 54-58% accuracy without it), by bits as the linear_warmup it replaces.
- CosineDecay(rate, steps, floor): rate to floor over steps by half a cosine, then floor (SGDR's
  eq. (5) without restarts, Loshchilov & Hutter 2017; D3's grouping and math.cos).
- LinearDecay(rate, steps, floor): rate to floor linearly over steps, then floor.

Warmup(steps, CosineDecay(rate, total - steps, rate / 10)) is the production recipe's schedule
(GPT-3; nanoGPT's min_lr = lr / 10). The trainer reads a schedule in pure Python, so all three
implementations train at the same rate bits.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


def _check_rate(name: str, value: float) -> None:
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and at least 0; got {value!r}")


def _check_steps(steps: int) -> None:
    if steps < 1:
        raise ValueError(f"steps must be at least 1; got {steps!r}")


@dataclass(frozen=True)
class Constant:
    rate: float

    def __post_init__(self) -> None:
        _check_rate("rate", self.rate)

    def __call__(self, step: int) -> float:
        return self.rate


@dataclass(frozen=True)
class Warmup:
    steps: int
    then: Schedule

    def __post_init__(self) -> None:
        _check_steps(self.steps)

    def __call__(self, step: int) -> float:
        if step < self.steps:
            # (step + 1): step 0 gets then(0) / steps rather than a wasted 0.0
            return self.then(0) * ((step + 1) / self.steps)
        return self.then(step - self.steps)


@dataclass(frozen=True)
class CosineDecay:
    rate: float
    steps: int
    floor: float

    def __post_init__(self) -> None:
        _check_rate("rate", self.rate)
        _check_rate("floor", self.floor)
        _check_steps(self.steps)
        if self.floor > self.rate:
            raise ValueError(f"floor {self.floor!r} is above rate {self.rate!r}")

    def __call__(self, step: int) -> float:
        if step >= self.steps:
            return self.floor
        return self.floor + (self.rate - self.floor) * (1.0 + math.cos(math.pi * step / self.steps)) / 2.0


@dataclass(frozen=True)
class LinearDecay:
    rate: float
    steps: int
    floor: float

    def __post_init__(self) -> None:
        _check_rate("rate", self.rate)
        _check_rate("floor", self.floor)
        _check_steps(self.steps)
        if self.floor > self.rate:
            raise ValueError(f"floor {self.floor!r} is above rate {self.rate!r}")

    def __call__(self, step: int) -> float:
        if step >= self.steps:
            return self.floor
        return self.floor + (self.rate - self.floor) * (1.0 - step / self.steps)


type Schedule = Constant | Warmup | CosineDecay | LinearDecay

# what the trainer's learning_rate takes (D4): a float means Constant; a bare callable is still
# accepted but can't be recorded in a run file (D5)
type LearningRate = float | Schedule | Callable[[int], float]
