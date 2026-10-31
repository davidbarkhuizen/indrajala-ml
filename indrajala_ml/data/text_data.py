"""
A character-level text corpus as next-token examples (the sequence task workplan, D2-D4): Tiny
Shakespeare, Herodotus in Rawlinson's translation, Ibn Khaldun's Muqaddimah in Arabic, or Euclid's
Elements in Heath's translation, each fetched and checksum-verified by scripts/data/fetch_datasets.py
and read as UTF-8 (all but the Muqaddimah are ASCII).

The vocabulary is the corpus's characters, sorted, each a token id by its place. The first
TRAIN_FRACTION of the text is for training and the rest held out (nanoGPT's split), each part cut
into non-overlapping windows of context + 1 characters from its start, a remainder shorter than a
window dropped. A window is one example: its first context ids the state, as floats (State is a
tuple of floats; ids are exact as floats), and its last context ids the labels, each the id that
follows the state's id at the same position.

The spread split (the attention-dropout workplan, D8) holds out text from across the corpus
instead: the text cut into SPREAD_BLOCKS equal blocks, each block's first TRAIN_FRACTION for
training and the rest held out, each piece cut into windows on its own, so no window crosses from
one piece into the next. A gap that the contiguous split shows and the spread one doesn't is the
text's shift from its start to its end, not overfitting.

Packed windows (the segments workplan, D2, D7; load_packed_text_dataset): the text's paragraphs
(paragraphs(): the pieces between runs of blank lines) in order, each followed by the separator
token (id len(vocabulary), SEPARATOR; the pad, PAD, is id len(vocabulary) + 1 and never appears in
training), concatenated and cut into windows as above. Both splits cut at paragraph boundaries, so
no paragraph is on both sides: the contiguous split's training part is the first paragraphs, cut at
the boundary nearest TRAIN_FRACTION of the characters, and the spread split's blocks are runs of whole paragraphs, each
split the same way, each piece packed on its own. The fixed-window loader stays, for the earlier
studies' reproducibility.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from indrajala_ml.model.protocols.classifier_protocols import Example

CORPORA = {
    "tinyshakespeare": "data/tinyshakespeare/tinyshakespeare.txt",
    "herodotus-rawlinson": "data/herodotus-rawlinson/herodotus-rawlinson.txt",
    "muqaddimah": "data/muqaddimah/muqaddimah.txt",
    "euclid-heath": "data/euclid-heath/euclid-heath.txt",
}
TEXT_PATH = CORPORA["tinyshakespeare"]
CONTEXT = 64
TRAIN_FRACTION = 0.9
SPREAD_BLOCKS = 100

# how load_text_dataset holds text out: the last part (nanoGPT's), or each block's last part
Split = Literal["contiguous", "spread"]

# a window's labels: the next token's id at each position
Labels = tuple[int, ...]

# the special tokens after a vocabulary's symbols, in id order (the segments workplan, D2): the
# separator ending each paragraph, then the pad
SEPARATOR = "<|sep|>"
PAD = "<|pad|>"
SPECIAL_TOKENS = (SEPARATOR, PAD)
# a paragraph break: a newline, then one or more lines holding only spaces or tabs
_PARAGRAPH_BREAK = re.compile(r"\n(?:[ \t]*\n)+")


@dataclass(frozen=True)
class Vocabulary:
    """The corpus's characters, sorted: a character's token id is its index in symbols."""

    symbols: str

    def __post_init__(self) -> None:
        assert list(self.symbols) == sorted(set(self.symbols)), "a vocabulary's symbols are distinct and sorted"

    @classmethod
    def of(cls, text: str) -> Vocabulary:
        return cls("".join(sorted(set(text))))

    def __len__(self) -> int:
        return len(self.symbols)

    def encode(self, text: str) -> list[int]:
        index = {symbol: i for i, symbol in enumerate(self.symbols)}
        return [index[symbol] for symbol in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.symbols[i] for i in ids)


def windows(ids: Sequence[int], context: int = CONTEXT) -> list[Example[Labels]]:
    """ids cut into non-overlapping windows of context + 1 from the start, each as an example."""
    assert context >= 1, f"context must be at least 1; got {context}"
    size = context + 1
    return [
        (tuple(float(i) for i in ids[start : start + context]), tuple(ids[start + 1 : start + size]))
        for start in range(0, len(ids) - size + 1, size)
    ]


def split(text: str, train_fraction: float = TRAIN_FRACTION) -> tuple[str, str]:
    """The first train_fraction of text, and the rest."""
    assert 0.0 < train_fraction < 1.0, f"train_fraction is in (0, 1); got {train_fraction}"
    cut = int(train_fraction * len(text))
    return text[:cut], text[cut:]


def spread_split(
    text: str, train_fraction: float = TRAIN_FRACTION, blocks: int = SPREAD_BLOCKS
) -> tuple[list[str], list[str]]:
    """
    text cut into blocks equal blocks (block b is characters b * n // blocks up to (b + 1) * n //
    blocks), each split as split splits a text: the training pieces and the held-out pieces, in
    the text's order.
    """
    assert blocks >= 1 and len(text) >= blocks, f"a spread split needs a character per block; got {blocks} blocks"
    bounds = [b * len(text) // blocks for b in range(blocks + 1)]
    pieces = [split(text[start:end], train_fraction) for start, end in pairwise(bounds)]
    return [train for train, _ in pieces], [held_out for _, held_out in pieces]


def separator_id(vocabulary: Vocabulary) -> int:
    return len(vocabulary)


def pad_id(vocabulary: Vocabulary) -> int:
    return len(vocabulary) + 1


def paragraphs(text: str) -> list[str]:
    """text's paragraphs, in order: the pieces between runs of blank lines, none empty."""
    return [piece for piece in _PARAGRAPH_BREAK.split(text.strip("\n")) if piece.strip()]


def pack(pieces: Sequence[str], vocabulary: Vocabulary) -> list[int]:
    """pieces' ids, each paragraph followed by the separator."""
    separator = separator_id(vocabulary)
    return [i for piece in pieces for i in (*vocabulary.encode(piece), separator)]


def _nearest_cut(pieces: Sequence[str], target: float) -> int:
    # the count of leading paragraphs whose characters (separators counted) come nearest target; a
    # tie trains
    best_k, best_distance, seen = 0, target, 0
    for k, piece in enumerate(pieces, start=1):
        seen += len(piece) + 1
        if abs(seen - target) <= best_distance:
            best_k, best_distance = k, abs(seen - target)
    return best_k


def split_paragraphs(pieces: Sequence[str], train_fraction: float = TRAIN_FRACTION) -> tuple[list[str], list[str]]:
    """
    pieces cut at the paragraph boundary nearest train_fraction of their characters (separators
    counted; a tie trains): the first paragraphs, and the rest. A lone paragraph trains.
    """
    assert 0.0 < train_fraction < 1.0, f"train_fraction is in (0, 1); got {train_fraction}"
    k = _nearest_cut(pieces, train_fraction * sum(len(piece) + 1 for piece in pieces))
    return list(pieces[:k]), list(pieces[k:])


def spread_split_paragraphs(
    pieces: Sequence[str], train_fraction: float = TRAIN_FRACTION, blocks: int = SPREAD_BLOCKS
) -> tuple[list[list[str]], list[list[str]]]:
    """
    pieces in blocks of whole paragraphs, a paragraph in block b when it starts in the text's b-th
    of blocks equal parts (by characters, separators counted), each block's first paragraphs
    training and the rest held out; the training and the held-out pieces per block, empty ones
    left out.

    Each block is cut at the boundary nearest its share of the training characters, the rounding
    carried into the next block (error diffusion): a block holds only a few paragraphs of a
    long-form corpus (5 on the Muqaddimah), so rounding each alone toward the nearer boundary held
    out 4% where the fraction asks 10%; carried, the whole split holds out its fraction.
    """
    assert 0.0 < train_fraction < 1.0, f"train_fraction is in (0, 1); got {train_fraction}"
    total = sum(len(piece) + 1 for piece in pieces)
    grouped: list[list[str]] = [[] for _ in range(blocks)]
    start = 0
    for piece in pieces:
        grouped[min(start * blocks // total, blocks - 1)].append(piece)
        start += len(piece) + 1
    train: list[list[str]] = []
    held_out: list[list[str]] = []
    carried = 0.0  # training characters asked for so far and not yet given
    for block in grouped:
        if not block:
            continue
        size = sum(len(piece) + 1 for piece in block)
        target = train_fraction * size + carried
        k = _nearest_cut(block, target)
        carried = target - sum(len(piece) + 1 for piece in block[:k])
        if k:
            train.append(block[:k])
        if k < len(block):
            held_out.append(block[k:])
    return train, held_out


def load_packed_text_dataset(
    path: str = TEXT_PATH,
    context: int = CONTEXT,
    train_fraction: float = TRAIN_FRACTION,
    split_by: Split = "contiguous",
) -> tuple[list[Example[Labels]], list[Example[Labels]], Vocabulary]:
    """
    The packed training windows, the packed held-out windows and the vocabulary (of the whole
    text): paragraphs each followed by the separator (id len(vocabulary)), cut into windows, held
    out at a paragraph boundary, contiguously or spread.
    """
    text = load_text(path)
    vocabulary = Vocabulary.of(text)
    pieces = paragraphs(text)
    if split_by == "contiguous":
        train, held_out = split_paragraphs(pieces, train_fraction)
        return windows(pack(train, vocabulary), context), windows(pack(held_out, vocabulary), context), vocabulary
    assert split_by == "spread", f"a split is contiguous or spread; got {split_by!r}"
    train_blocks, held_out_blocks = spread_split_paragraphs(pieces, train_fraction)

    def cut(blocks: list[list[str]]) -> list[Example[Labels]]:
        return [window for block in blocks for window in windows(pack(block, vocabulary), context)]

    return cut(train_blocks), cut(held_out_blocks), vocabulary


def load_text(path: str = TEXT_PATH) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_text_dataset(
    path: str = TEXT_PATH,
    context: int = CONTEXT,
    train_fraction: float = TRAIN_FRACTION,
    split_by: Split = "contiguous",
) -> tuple[list[Example[Labels]], list[Example[Labels]], Vocabulary]:
    """
    The training windows, the held-out windows and the vocabulary (of the whole text), held out
    contiguously (the last part) or spread (each block's last part, spread_split).
    """
    text = load_text(path)
    vocabulary = Vocabulary.of(text)
    if split_by == "contiguous":
        train, held_out = split(text, train_fraction)
        return windows(vocabulary.encode(train), context), windows(vocabulary.encode(held_out), context), vocabulary
    assert split_by == "spread", f"a split is contiguous or spread; got {split_by!r}"
    train_pieces, held_out_pieces = spread_split(text, train_fraction)

    def cut(pieces: list[str]) -> list[Example[Labels]]:
        return [window for piece in pieces for window in windows(vocabulary.encode(piece), context)]

    return cut(train_pieces), cut(held_out_pieces), vocabulary
