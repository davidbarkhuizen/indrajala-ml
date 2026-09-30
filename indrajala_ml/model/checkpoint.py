"""
A network's checkpoint (docs/composable-layers-workplan.md, stage 5): its weights, as snapshot()
gives them, and its optimizer's state, as the optimizer's state() gives it. Restoring one puts the
network back where training was, the optimizer included, so training on from it takes the same
steps as training on from the moment it was taken. train.py's pocket restores the best epoch's.

Backend-free, so the pure-Python networks import it too. A checkpoint holds copies: later steps
don't move it, and restoring it doesn't alias it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerState[S]:
    """
    An optimizer's step count t and its update rule's state per layer (momentum's velocities,
    Adam's m and v), keyed by the layer's index as the optimizer keys it. A layer not yet stepped,
    or stepped by a rule without state (SGD, weight decay), has no entry.
    """

    t: int
    layers: dict[int, S]


@dataclass(frozen=True)
class Checkpoint[W, S]:
    """A network's weights (its snapshot()) and its optimizer's state."""

    weights: W
    optimizer: OptimizerState[S]
