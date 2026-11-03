"""
Greek-English parallel pairs packed into windows (roadmap step 15, the cross-attention workplan,
D4, D11, D13): the histories and Plutarch's Lives from indrajala-datasets-perseus-parallel,
fetched and checksum-verified by scripts/data/fetch_datasets.py, one pair a line as JSON
(`{work, ref, split, grc, eng}`), each split (train, validation, test) by the dataset's held-out
books and Lives.

Two character vocabularies, the source's (the Greek's characters, accents kept: D5) and the
target's (the English's), each of every split's text, as a text corpus's vocabulary is of the
whole text, and each with step 11's special tokens after its symbols: the separator (id
len(vocabulary)) and the pad (len(vocabulary) + 1).

A window holds whole pairs, in order, never a cut one (D11). Its source is each pair's Greek
followed by the separator, so a separator ends its pair as it ends a paragraph in step 11's packing
(`text_data.pack`). Its target row is the separator, then each pair's English followed by the
separator (D13). The decoder reads the row less its last token and predicts the row less its first
(step 9's shift), so a separator ends one pair's translation and starts the next. Pads stand before
a row's first token (left padding, step 11's D8): the source is window ids long, the target row
window + 1. A target whose input is a pad doesn't count (step 11's D6), so the first separator, the
start token, is given and never predicted.

Pair k is segment k of both sides: on the source by step 11's rule (a separator ends its segment),
on the target by D13's (a separator opens the segment after it). The model draws both from the ids
(stage 3). Pairs are packed greedily: a window takes the next pair while both sides fit. A pair that
doesn't fit an empty window is dropped, never cut, and the dataset lists it (`dropped`).

Training windows are packed anew every epoch from the training pairs, shuffled by the trainer's
generator (PackedPairEpochs, as text_data.PackedEpochs packs paragraphs), so the pairs sharing a
window change. The count of windows can change too, since greedy packing depends on the order. The
validation and test windows are packed once, in order.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Literal

from indrajala_ml.data.text_data import Vocabulary, pad_id, separator_id

PARALLEL_CORPORA = {
    "histories": "data/perseus-parallel/histories.jsonl",
    "lives": "data/perseus-parallel/lives.jsonl",
}
# the window on both sides (D11): step 12's longest
WINDOW = 1024

Split = Literal["train", "validation", "test"]
SPLITS: tuple[Split, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class Pair:
    """One citation unit in both languages: its work, its Perseus citation, its split and its texts."""

    work: str
    ref: str
    split: Split
    grc: str
    eng: str


def load_pairs(path: str) -> list[Pair]:
    """A parallel corpus's pairs, in text order."""
    pairs: list[Pair] = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            record = json.loads(line)
            assert record["split"] in SPLITS, f"{path}:{number}: a split is one of {SPLITS}; got {record['split']!r}"
            assert record["grc"] and record["eng"], f"{path}:{number}: a pair has both texts"
            pairs.append(Pair(record["work"], record["ref"], record["split"], record["grc"], record["eng"]))
    return pairs


@dataclass(frozen=True)
class PairWindow:
    """
    One window of whole pairs (D11, D13): the source's ids (window long) and the target row's
    (window + 1), each left-padded, and the count of pairs it holds.
    """

    source: tuple[int, ...]
    target: tuple[int, ...]
    pairs: int

    @property
    def decoder_input(self) -> tuple[int, ...]:
        """The target row less its last token: what the decoder reads."""
        return self.target[:-1]

    @property
    def labels(self) -> tuple[int, ...]:
        """The target row less its first token: what the decoder predicts at each position."""
        return self.target[1:]


def source_length(pair: Pair) -> int:
    """The pair's Greek and its separator."""
    return len(pair.grc) + 1


def target_length(pair: Pair) -> int:
    """The pair's English and its separator (the row's opening separator is the window's, not the pair's)."""
    return len(pair.eng) + 1


def fits(pair: Pair, window: int = WINDOW) -> bool:
    """Whether the pair fits an empty window: its source in window ids, its target row in window + 1."""
    return source_length(pair) <= window and 1 + target_length(pair) <= window + 1


def pack_pairs(pairs: Sequence[Pair], source: Vocabulary, target: Vocabulary, window: int = WINDOW) -> list[PairWindow]:
    """pairs in order, packed greedily into windows: the next pair joins a window while both sides fit."""
    assert window >= 2, f"a window holds a pair of one character a side; got {window}"
    assert all(fits(pair, window) for pair in pairs), "every pair packed fits an empty window: drop the rest first"
    windows: list[PairWindow] = []
    group: list[Pair] = []
    used_source = used_target = 0
    for pair in pairs:
        if group and (used_source + source_length(pair) > window or used_target + target_length(pair) > window):
            windows.append(_window(group, source, target, window))
            group, used_source, used_target = [], 0, 0
        group.append(pair)
        used_source += source_length(pair)
        used_target += target_length(pair)
    if group:
        windows.append(_window(group, source, target, window))
    return windows


def _window(group: Sequence[Pair], source: Vocabulary, target: Vocabulary, window: int) -> PairWindow:
    source_ids = [i for pair in group for i in (*source.encode(pair.grc), separator_id(source))]
    target_ids = [separator_id(target)] + [
        i for pair in group for i in (*target.encode(pair.eng), separator_id(target))
    ]
    return PairWindow(
        source=(pad_id(source),) * (window - len(source_ids)) + tuple(source_ids),
        target=(pad_id(target),) * (window + 1 - len(target_ids)) + tuple(target_ids),
        pairs=len(group),
    )


@dataclass(frozen=True)
class PackedPairEpochs:
    """
    Training windows packed anew every epoch: the training pairs shuffled by the trainer's
    generator, then packed (pack_pairs), so every pair is seen once an epoch while the pairs
    sharing a window change. A run resumed from its checkpoint, whose generator state is the epoch
    boundary's, packs the same windows.
    """

    pairs: tuple[Pair, ...]
    source: Vocabulary
    target: Vocabulary
    window: int = WINDOW

    def epoch(self, rng: Random) -> list[PairWindow]:
        order = list(self.pairs)
        rng.shuffle(order)
        return pack_pairs(order, self.source, self.target, self.window)


@dataclass(frozen=True)
class ParallelDataset:
    """A parallel corpus in windows: training packed every epoch, validation and test once, in order."""

    train: PackedPairEpochs
    validation: list[PairWindow]
    test: list[PairWindow]
    source: Vocabulary
    target: Vocabulary
    dropped: tuple[Pair, ...]  # the pairs longer than a window, every split's


def load_parallel_dataset(corpus: str = "histories", window: int = WINDOW, path: str | None = None) -> ParallelDataset:
    """corpus's pairs (PARALLEL_CORPORA, or the file at path) in windows of window ids a side."""
    pairs = load_pairs(path if path is not None else PARALLEL_CORPORA[corpus])
    source = Vocabulary.of("".join(pair.grc for pair in pairs))
    target = Vocabulary.of("".join(pair.eng for pair in pairs))
    kept = [pair for pair in pairs if fits(pair, window)]
    dropped = tuple(pair for pair in pairs if not fits(pair, window))

    def of(split: Split) -> list[Pair]:
        return [pair for pair in kept if pair.split == split]

    return ParallelDataset(
        train=PackedPairEpochs(tuple(of("train")), source, target, window),
        validation=pack_pairs(of("validation"), source, target, window),
        test=pack_pairs(of("test"), source, target, window),
        source=source,
        target=target,
        dropped=dropped,
    )
