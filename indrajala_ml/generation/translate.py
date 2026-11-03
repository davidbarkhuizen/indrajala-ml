"""
Translating with an encoder-decoder (the cross-attention workplan, D7, D13, D15): a network's
decoder(sources) holds the sources encoded once, and translate decodes each row from one separator,
the target's start (D13), drawing a token a row per step until the row draws the next separator,
its end, or reaches the length cap. A row that has ended still steps, its tokens discarded, so the
batch stays in step; its draws still take one value each from the generator.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from indrajala_ml.generation.sampling import Sampler, UniformSource


def translate(
    decoder: Any, rows: int, separator: int, length: int, sampler: Sampler, rng: UniformSource
) -> list[list[int]]:
    """rows translations, each up to length ids, the separators that open and end them left out."""
    assert rows >= 1 and length >= 1, f"a translation has at least one row and one token; got {rows} and {length}"
    probabilities: Sequence[Sequence[float]] = decoder.prefill_batch([[separator]] * rows)
    out: list[list[int]] = [[] for _ in range(rows)]
    ended = [False] * rows
    for i in range(length):
        tokens = [sampler(p, rng) for p in probabilities]
        for row, token in enumerate(tokens):
            if ended[row]:
                continue
            if token == separator:
                ended[row] = True
            else:
                out[row].append(token)
        if all(ended) or i + 1 == length:
            break
        probabilities = decoder.step_batch(tokens)
    return out
