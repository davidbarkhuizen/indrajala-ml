from __future__ import annotations

import math

from indrajala_ml.pcg64 import Pcg64Generator


def fan_in_aware_weights_and_bias(rng: Pcg64Generator, fan_in: int) -> tuple[list[float], float]:
    """
    fan_in weights and a bias drawn from rng uniformly from [-limit, limit], limit =
    1/sqrt(fan_in): the formula every pure-Python fan-in-aware initialization uses (BackpropLayer's
    and ConvKernel's randomize_fan_in_aware, and so randomize_fan_in_aware(network)). Its own
    module, since the layers and the network base that builds them both need it.
    """
    limit = 1.0 / math.sqrt(fan_in)
    weights = [rng.uniform(-limit, limit) for _ in range(fan_in)]
    bias = rng.uniform(-limit, limit)
    return weights, bias


def fan_in_aware_weights(rng: Pcg64Generator, fan_in: int) -> list[float]:
    """fan_in_aware_weights_and_bias's weights, without drawing a bias: a linear layer's node, which
    has none (the batch-norm workplan, D2)."""
    limit = 1.0 / math.sqrt(fan_in)
    return [rng.uniform(-limit, limit) for _ in range(fan_in)]
