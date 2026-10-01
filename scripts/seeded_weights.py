"""
Seeded starting weights for the timing scripts that scripts/ab.py runs from one tree against both
sides of a change, so they must run on a tree from before the RNG generators workplan's stage 3
too: there a network draws from np.random's global state, and since then from its own generator.

    from seeded_weights import seeded_randomized
"""

import inspect
from typing import Any

import numpy as np


def seeded_randomized(cls: Any, seed: int, *args: Any) -> Any:
    """cls.randomized(*args) from seed: its own generator where randomized takes seed=, np.random before."""
    if "seed" in inspect.signature(cls.randomized).parameters:
        return cls.randomized(*args, seed=seed)
    np.random.seed(seed)
    return cls.randomized(*args)
