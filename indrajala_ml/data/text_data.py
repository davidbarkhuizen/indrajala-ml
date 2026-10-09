"""
A character-level text corpus as next-token examples (the sequence task workplan, D2-D4): Tiny
Shakespeare, fetched and checksum-verified by scripts/fetch_datasets.py.

The vocabulary is the corpus's characters, sorted, each a token id by its place. The first
TRAIN_FRACTION of the text is for training and the rest held out (nanoGPT's split), each part cut
into non-overlapping windows of context + 1 characters from its start, a remainder shorter than a
window dropped. A window is one example: its first context ids the state, as floats (State is a
tuple of floats; ids are exact as floats), and its last context ids the labels, each the id that
follows the state's id at the same position.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from indrajala_ml.model.protocols.classifier_protocols import Example

TEXT_PATH = "data/tinyshakespeare/tinyshakespeare.txt"
CONTEXT = 64
TRAIN_FRACTION = 0.9

# a window's labels: the next token's id at each position
Labels = tuple[int, ...]


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


def load_text(path: str = TEXT_PATH) -> str:
    with open(path, encoding="ascii") as f:
        return f.read()


def load_text_dataset(
    path: str = TEXT_PATH, context: int = CONTEXT, train_fraction: float = TRAIN_FRACTION
) -> tuple[list[Example[Labels]], list[Example[Labels]], Vocabulary]:
    """The training windows, the held-out windows and the vocabulary (of the whole text)."""
    text = load_text(path)
    vocabulary = Vocabulary.of(text)
    train, held_out = split(text, train_fraction)
    return windows(vocabulary.encode(train), context), windows(vocabulary.encode(held_out), context), vocabulary
