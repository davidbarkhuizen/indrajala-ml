from __future__ import annotations

from typing import Callable


def linear_warmup(target_rate: float, warmup_steps: int) -> Callable[[int], float]:
    """
    A schedule (step index -> learning rate) ramping linearly from target_rate/warmup_steps to
    target_rate over warmup_steps steps, then holding: the warmup Goyal et al. 2017 pair with the
    linear scaling rule, since a scaled rate from step one is unstable (batch 128's rate diverged to
    54-58% accuracy without it).

    It uses step + 1, so step 0 gets target_rate/warmup_steps rather than a wasted 0.0.
    """

    def schedule(step: int) -> float:
        return target_rate * min((step + 1) / warmup_steps, 1.0)

    return schedule
