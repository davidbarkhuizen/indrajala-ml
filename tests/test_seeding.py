import random

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.seeding import seed_everything


def _draws() -> tuple[float, list[float], list[float]]:
    return random.random(), np.random.random(5).tolist(), pa.random(5).tolist()


def test_seed_everything_seeds_random_numpy_and_the_crate():
    seed_everything(11)
    first = _draws()
    random.seed(11)
    np.random.seed(11)
    pa.seed(11)
    assert _draws() == first
    # the crate's stream is numpy's, seeded alike
    assert first[2] == first[1]


def test_seed_everything_none_reseeds_from_entropy():
    seed_everything(None)
    first = _draws()
    seed_everything(None)
    assert _draws() != first
