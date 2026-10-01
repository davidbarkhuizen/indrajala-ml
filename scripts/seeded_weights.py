"""
Seeded starting weights and shuffles for the timing scripts that scripts/ab.py runs from one tree
against both sides of a change, so they must run on a tree from before the RNG generators
workplan's stages 3 and 5 too: there a network draws from np.random's global state, and a trainer
shuffles from random's; since then each draws from its own generator.

    from seeded_weights import seeded_randomized, seeded_shuffle
"""

import inspect
import random
from collections.abc import Callable
from typing import Any

import numpy as np


def seeded_randomized(cls: Any, seed: int, *args: Any) -> Any:
    """cls.randomized(*args) from seed: its own generator where randomized takes seed=, np.random before."""
    if "seed" in inspect.signature(cls.randomized).parameters:
        return cls.randomized(*args, seed=seed)
    np.random.seed(seed)
    return cls.randomized(*args)


def seeded_shuffle(trainer: Callable[..., Any], seed: int) -> dict[str, Any]:
    """
    The keyword arguments that seed trainer's shuffle: its own random.Random where it takes rng=,
    after seeding the global random before. Either gives the same order.
    """
    if "rng" in inspect.signature(trainer).parameters:
        return {"rng": random.Random(seed)}
    random.seed(seed)
    return {}
