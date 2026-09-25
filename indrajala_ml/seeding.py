from __future__ import annotations

import random

import indrajala_math_rust as pa
import numpy as np


def seed_everything(seed: int | None) -> None:
    """
    Seeds every global RNG the package draws from, alike: Python's random (the per-node networks
    and the trainers' shuffles), np.random (the numpy backend's weights and dropout masks) and
    the crate's (the Rust backend's, numpy's stream in a separate state). None reseeds each from
    OS entropy. A forked worker inherits all three states, so it calls this before it draws.
    """
    random.seed(seed)
    np.random.seed(seed)
    pa.seed(seed)
