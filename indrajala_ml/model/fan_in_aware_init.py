from __future__ import annotations

import math
import random


def fan_in_aware_weights_and_bias(fan_in: int) -> tuple[list[float], float]:
    """
    fan_in weights and a bias drawn uniformly from [-limit, limit], limit = 1/sqrt(fan_in): the
    formula every pure-Python fan-in-aware initialization uses (BackpropLayer's and ConvKernel's
    randomize_fan_in_aware, and so randomize_fan_in_aware(network)). Its own module, since the
    layers and the network base that builds them both need it.
    """
    limit = 1.0 / math.sqrt(fan_in)
    weights = [random.uniform(-limit, limit) for _ in range(fan_in)]
    bias = random.uniform(-limit, limit)
    return weights, bias
