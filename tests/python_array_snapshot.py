"""
A pure-Python network against its array twin: its snapshot in the array networks' shape, and the
twin seeded as it was. Pure Python draws from numpy's PCG64 stream in numpy's order (the RNG
draw-order workplan, D2 and D3), so from one seed both start from the same weights by bits, with
their generators in the same state, and a parity test seeds both sides rather than restoring one
from the other (D4).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from indrajala_ml.model.layers.python.attention_layer import AttentionLayer
from indrajala_ml.model.layers.python.batch_norm_layer import BatchNormLayer
from indrajala_ml.model.layers.python.layer_norm_layer import LayerNormLayer
from indrajala_ml.model.layers.python.linear_conv_layer import LinearConvLayer
from indrajala_ml.model.layers.python.linear_layer import LinearLayer
from indrajala_ml.pcg64 import generator_state


def projections(layer: AttentionLayer, entry: list[Any]) -> list[list[Any]]:
    """An attention layer's weight sets split into Wq's, Wk's, Wv's (h * d_k rows each) and Wo's (d)."""
    w = layer.width
    return [entry[:w], entry[w : 2 * w], entry[2 * w : 3 * w], entry[3 * w :]]


def as_array_snapshot(python: Any) -> list[tuple[Any, ...]]:
    """
    The pure-Python snapshot, per node, unit or feature, as the array networks', per parameter: a
    batch norm's (gamma, beta, mean, var), a layer norm's (gamma, beta), attention's (Wq, bq, Wk,
    bk, Wv, bv, Wo, bo), a layer without a bias as (W,), and every other layer as (W, b).
    """
    snapshot: list[tuple[Any, ...]] = []
    for layer, entry in zip(python.trainable_layers, python.snapshot(), strict=True):
        if isinstance(layer, BatchNormLayer):
            gamma, beta, mean, var = zip(*entry)
            snapshot.append(([g for (g,) in gamma], list(beta), list(mean), list(var)))
        elif isinstance(layer, LayerNormLayer):
            snapshot.append(([gamma for (gamma,), _ in entry], [beta for _, beta in entry]))
        elif isinstance(layer, AttentionLayer):
            snapshot.append(
                tuple(
                    values
                    for rows in projections(layer, entry)
                    for values in ([w for w, _ in rows], [b for _, b in rows])
                )
            )
        elif isinstance(layer, LinearLayer | LinearConvLayer):
            snapshot.append(([weights for (weights,) in entry],))
        else:
            snapshot.append(tuple(list(values) for values in zip(*entry)))
    return snapshot


def assert_seeded_alike(python: Any, array: Any) -> None:
    """python and array, each randomized from the same seed, hold the same weights by bits, and
    their generators the same state."""
    expected = [[np.asarray(values, dtype=np.float64).tobytes() for values in entry] for entry in array.snapshot()]
    actual = [
        [np.asarray(values, dtype=np.float64).tobytes() for values in entry] for entry in as_array_snapshot(python)
    ]
    assert actual == expected
    assert generator_state(python.rng) == generator_state(array.rng)


def seeded_like[T](python: Any, array: T, seed: int) -> T:
    """array randomized from seed, the seed python was randomized from, and checked to match it
    (assert_seeded_alike)."""
    twin: Any = array
    twin.rng = twin.backend.default_rng(seed)
    twin.randomize()
    assert_seeded_alike(python, twin)
    return array
