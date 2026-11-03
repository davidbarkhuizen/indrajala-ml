"""
An encoder-decoder's evaluation on held-out pairs (the cross-attention workplan, D6, D9): bits per
character under teacher forcing over the split's packed windows, the targets that count (not those
whose input is a pad, so the windows' first separator, the start token, is given), and chrF of the
greedy and the sampled translations of each pair's Greek alone (D11: a decode encodes one source a
row, its mask padding only), each against the pair's English.

A translation decodes from one separator to the next, or to the cap: the longest English the
target window holds (window - 1 tokens, the opening separator's position taken). Its ids are the
target vocabulary's characters; a separator is never among them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from indrajala_ml.data.parallel_data import Pair, PairWindow
from indrajala_ml.data.text_data import Vocabulary, pad_id, separator_id
from indrajala_ml.generation.chrf import chrf
from indrajala_ml.generation.sampling import Sampler, UniformSource
from indrajala_ml.generation.translate import translate


@dataclass(frozen=True)
class TranslationEvaluation:
    """Teacher-forced cross-entropy in nats per counted character, and chrF of greedy and sampled translations."""

    cross_entropy: float
    characters: int
    chrf_greedy: float
    chrf_sampled: float
    pairs: int

    @property
    def bits_per_character(self) -> float:
        return self.cross_entropy / math.log(2.0)


def teacher_forced(network: Any, windows: Sequence[PairWindow], pad: int) -> tuple[float, int]:
    """The mean -log probability of each counted label, in nats, and the count, over the windows."""
    assert windows, "windows must not be empty"
    total, count = 0.0, 0
    for window in windows:
        state = tuple(float(i) for i in (*window.source, *window.decoder_input))
        probabilities = network.predict_probabilities(state)
        for t, label in enumerate(window.labels):
            if window.decoder_input[t] == pad:
                continue
            p = float(probabilities[t][label])
            total += math.inf if p == 0.0 else -math.log(p)
            count += 1
    return total / count, count


def translations(
    network: Any,
    pairs: Sequence[Pair],
    source: Vocabulary,
    target: Vocabulary,
    sampler: Sampler,
    rng: UniformSource,
    batch: int = 8,
) -> list[str]:
    """Each pair's Greek translated, batch rows a decode, as English text."""
    cap = network.target_tokens - 1
    out: list[str] = []
    for start in range(0, len(pairs), batch):
        group = pairs[start : start + batch]
        sources = [[*source.encode(pair.grc), separator_id(source)] for pair in group]
        decoder = network.decoder(sources)
        for ids in translate(decoder, len(group), separator_id(target), cap, sampler, rng):
            out.append(target.decode(ids))
    return out


def translation_evaluate(
    network: Any,
    windows: Sequence[PairWindow],
    pairs: Sequence[Pair],
    source: Vocabulary,
    target: Vocabulary,
    rng: UniformSource,
    temperature: float = 1.0,
    batch: int = 8,
) -> TranslationEvaluation:
    """network on a split: its windows teacher-forced, and its pairs translated greedily and by sampling at temperature."""
    assert pairs, "pairs must not be empty"
    cross_entropy, characters = teacher_forced(network, windows, pad_id(target))
    references = [pair.eng for pair in pairs]
    greedy = translations(network, pairs, source, target, Sampler(temperature=0.0), rng, batch)
    sampled = translations(network, pairs, source, target, Sampler(temperature=temperature), rng, batch)
    return TranslationEvaluation(
        cross_entropy=cross_entropy,
        characters=characters,
        chrf_greedy=chrf(greedy, references),
        chrf_sampled=chrf(sampled, references),
        pairs=len(pairs),
    )
