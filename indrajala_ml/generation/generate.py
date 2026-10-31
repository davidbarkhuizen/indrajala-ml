"""
Sampling text from a sequence model (the generation workplan, D1 (b), D4): the prompt prefilled,
then one draw and one step per new token. A cached decode (Decoder) by default; the uncached one,
a full pass over the last window per token, is what the generation benchmark measures it against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from indrajala_ml.generation.sampling import Probabilities, Sampler, UniformSource


def generate(
    prefill: Callable[[Sequence[int]], Probabilities],
    step: Callable[[int], Probabilities],
    prompt: Sequence[int],
    length: int,
    sampler: Sampler,
    rng: UniformSource,
) -> list[int]:
    """length new ids after prompt: each drawn from the last probabilities, then stepped (but the last)."""
    assert length >= 0, f"a sample's length is at least 0; got {length}"
    p = prefill(prompt)
    ids: list[int] = []
    for i in range(length):
        ids.append(sampler(p, rng))
        if i + 1 < length:
            p = step(ids[-1])
    return ids


def cached(network: Any) -> tuple[Callable[[Sequence[int]], Probabilities], Callable[[int], Probabilities]]:
    """A network's decoder's prefill and step."""
    decoder = network.decoder()
    return decoder.prefill, decoder.step


def uncached(network: Any) -> tuple[Callable[[Sequence[int]], Probabilities], Callable[[int], Probabilities]]:
    """
    prefill and step as a full pass each over the last window of tokens (a prompt shorter than the
    window is the decoder's prefill, which a full pass can't take): every token's whole window again.
    """
    window = network.tokens
    history: list[int] = []
    decoder = network.decoder()

    def full(ids: Sequence[int]) -> Probabilities:
        if len(ids) < window:
            return decoder.prefill(ids)
        return network.predict_probabilities(tuple(float(i) for i in ids[-window:]))[-1]

    def prefill(ids: Sequence[int]) -> Probabilities:
        history[:] = list(ids)
        return full(history)

    def step(token: int) -> Probabilities:
        history.append(token)
        return full(history)

    return prefill, step
