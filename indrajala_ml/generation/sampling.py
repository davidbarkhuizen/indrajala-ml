"""
Sampling one token from a model's probabilities at one position (the generation workplan, D4): a
chain of filters, each a pure function from probabilities to probabilities, then one draw. A new
sampler is a new filter.

- temperature(t): a softmax of log p / t, max-shifted, a left fold for the sum; t = 0 is the
  argmax (the first most likely id); t = 1 passes p on unchanged.
- top_k(k): the k most likely ids, ties by id, the rest set to 0.
- top_p(p): the smallest set, in descending probability (ties by id), whose mass reaches p
  (Holtzman et al. 2019), the rest set to 0.

Each filter that changes p renormalizes what it keeps, dividing by a left fold of the kept values
in id order. The draw takes one uniform u in [0, 1) from the generator and returns the first id
whose cumulative probability, a left fold in id order, exceeds u. The cumulative sum's rounding
decides a draw at a boundary, so two implementations' probabilities, a last bit apart, can sample
different text from one seed (D7); one model file and one seed sample the same text.

Backend-free and numpy-free: probabilities are lists of floats, and the generator is any with
random() (indrajala_ml.pcg64's, numpy's or the crate's), seeded once per sample.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from indrajala_ml.model.layers.python.batch_norm_layer import fold

Probabilities = list[float]
Filter = Callable[[Sequence[float]], Probabilities]


class UniformSource(Protocol):
    def random(self) -> float: ...


def _renormalized(p: Sequence[float]) -> Probabilities:
    total = fold(p)
    assert total > 0.0, "a filter kept no probability mass"
    return [p_i / total for p_i in p]


def _by_rank(p: Sequence[float]) -> list[int]:
    # ids in descending probability, ties by id
    return sorted(range(len(p)), key=lambda i: (-p[i], i))


def argmax(p: Sequence[float]) -> int:
    """The first most likely id."""
    return _by_rank(p)[0]


def temperature(t: float) -> Filter:
    """softmax(log p / t), max-shifted; 0 the argmax as a one-hot, 1 the identity."""
    assert t >= 0.0, f"a temperature is at least 0; got {t}"

    def apply(p: Sequence[float]) -> Probabilities:
        if t == 0.0:
            best = argmax(p)
            return [1.0 if i == best else 0.0 for i in range(len(p))]
        if t == 1.0:
            return list(p)
        logits = [math.log(p_i) / t if p_i > 0.0 else -math.inf for p_i in p]
        m = max(logits)
        return _renormalized([math.exp(z - m) for z in logits])

    return apply


def top_k(k: int) -> Filter:
    """The k most likely ids, ties by id, renormalized."""
    assert k >= 1, f"top-k keeps at least one id; got {k}"

    def apply(p: Sequence[float]) -> Probabilities:
        kept = set(_by_rank(p)[:k])
        return _renormalized([p_i if i in kept else 0.0 for i, p_i in enumerate(p)])

    return apply


def top_p(mass: float) -> Filter:
    """The smallest set, in descending probability (ties by id), whose mass reaches mass, renormalized."""
    assert 0.0 < mass <= 1.0, f"top-p's mass is in (0, 1]; got {mass}"

    def apply(p: Sequence[float]) -> Probabilities:
        kept: set[int] = set()
        total = 0.0
        for i in _by_rank(p):
            kept.add(i)
            total += p[i]
            if total >= mass:
                break
        return _renormalized([p_i if i in kept else 0.0 for i, p_i in enumerate(p)])

    return apply


def draw(p: Sequence[float], u: float) -> int:
    """
    The first id whose cumulative probability, a left fold in id order, exceeds u, in [0, 1); the
    last id with any probability when rounding leaves the total at or below u.
    """
    total = 0.0
    last = -1
    for i, p_i in enumerate(p):
        if p_i > 0.0:
            total += p_i
            last = i
            if total > u:
                return i
    assert last >= 0, "nothing to draw: every probability is 0"
    return last


@dataclass(frozen=True)
class Sampler:
    """
    The chain temperature, top-k, top-p (each left out at its default), then one draw: the settings
    sample_text.py's sample takes. Temperature 0 is greedy decoding; its draw still takes one value
    from the generator, so a sample's draws stay one per token whatever the settings.
    """

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def filters(self) -> list[Filter]:
        chain: list[Filter] = []
        if self.temperature != 1.0:
            chain.append(temperature(self.temperature))
        if self.top_k is not None:
            chain.append(top_k(self.top_k))
        if self.top_p is not None and self.top_p < 1.0:
            chain.append(top_p(self.top_p))
        return chain

    def filtered(self, p: Sequence[float]) -> Probabilities:
        """p through the chain."""
        out = list(p)
        for apply in self.filters():
            out = apply(out)
        return out

    def __call__(self, p: Sequence[float], rng: UniformSource) -> int:
        """One id drawn from p through the chain, taking one uniform from rng."""
        return draw(self.filtered(p), rng.random())
