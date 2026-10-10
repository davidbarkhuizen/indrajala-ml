from __future__ import annotations

import math

from indrajala_ml.pcg64 import Pcg64Generator


def fan_in_aware_weights_and_biases(
    rng: Pcg64Generator, rows: int, fan_in: int, bias: bool = True
) -> tuple[list[list[float]], list[float]]:
    """
    A layer's rows weight rows of fan_in weights, then its rows biases, drawn from rng uniformly
    from [-limit, limit], limit = 1/sqrt(fan_in): every row's weights in row order, then every
    bias, numpy's order (array_layer.fan_in_aware_random_layer's W, then b), so a pure-Python
    layer seeded alike starts with the array twin's weights (the RNG draw-order workplan, D2). The
    formula every pure-Python fan-in-aware layer with a bias draws (dense, conv, token-wise dense,
    an attention projection), and so randomize_fan_in_aware(network). Without bias, the weights
    alone and zero biases: rows that have none (a linear conv's kernels, an embedding table). Its
    own module, since the layers and the network base that builds them both need it.
    """
    weights = fan_in_aware_weights(rng, rows, fan_in)
    if not bias:
        return weights, [0.0] * rows
    limit = 1.0 / math.sqrt(fan_in)
    return weights, [rng.uniform(-limit, limit) for _ in range(rows)]


def fan_in_aware_weights(rng: Pcg64Generator, rows: int, fan_in: int) -> list[list[float]]:
    """fan_in_aware_weights_and_biases's weight rows, without drawing a bias: a layer that has none
    (linear, linear conv, an embedding table; the batch-norm workplan, D2)."""
    limit = 1.0 / math.sqrt(fan_in)
    return [[rng.uniform(-limit, limit) for _ in range(fan_in)] for _ in range(rows)]
