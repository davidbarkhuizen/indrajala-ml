"""
A model's tokenizer file (the generation workplan, D5): `<model>.tokenizer.json` beside the model's
format-2 file, `{"kind": "characters", "corpus": ..., "sha256": ..., "symbols": ...}`, the corpus's
name, the checksum of the text the vocabulary was read from, and the symbols in id order. `kind`
admits another tokenizer (BPE, words) later without touching format 2.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from indrajala_ml.data.text_data import Vocabulary

KINDS = ("characters",)


def tokenizer_path(model: str | Path) -> Path:
    """The tokenizer file beside a model file: model.json's is model.tokenizer.json."""
    model = Path(model)
    return model.with_name(f"{model.stem}.tokenizer.json")


@dataclass(frozen=True)
class CharacterTokenizer:
    """A character vocabulary from a named corpus: a character's id is its index in symbols."""

    corpus: str
    sha256: str
    symbols: str

    kind = "characters"

    @classmethod
    def of(cls, corpus: str, text: str) -> CharacterTokenizer:
        """The vocabulary of text, the corpus named corpus, with text's checksum."""
        return cls(corpus, hashlib.sha256(text.encode()).hexdigest(), Vocabulary.of(text).symbols)

    @property
    def vocabulary(self) -> Vocabulary:
        return Vocabulary(self.symbols)

    def __len__(self) -> int:
        return len(self.symbols)

    def encode(self, text: str) -> list[int]:
        """text's ids, refused if it holds a character the corpus doesn't."""
        unknown = sorted(set(text) - set(self.symbols))
        if unknown:
            raise ValueError(f"the {self.corpus} vocabulary has no {''.join(unknown)!r}")
        return self.vocabulary.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self.vocabulary.decode(ids)

    def save(self, path: str | Path) -> None:
        state = {"kind": self.kind, "corpus": self.corpus, "sha256": self.sha256, "symbols": self.symbols}
        Path(path).write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("kind") not in KINDS:
            raise ValueError(f"{path}: an unknown tokenizer kind {state.get('kind')!r}; known: {', '.join(KINDS)}")
        return cls(state["corpus"], state["sha256"], state["symbols"])
