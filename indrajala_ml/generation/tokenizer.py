"""
A model's tokenizer file (the generation workplan, D5): `<model>.tokenizer.json` beside the model's
format-2 file, `{"kind": "characters", "corpus": ..., "sha256": ..., "symbols": ...}`, the corpus's
name, the checksum of the text the vocabulary was read from, and the symbols in id order. `kind`
admits another tokenizer (BPE, words) later without touching format 2.

A tokenizer of packed text (the segments workplan, D2) adds `"special": ["<|sep|>", "<|pad|>"]`,
the special tokens after the symbols in id order: the separator (id V, V the symbols' count), which
a model predicts and generation can emit, and the pad (V + 1), outside every class. A model over it
embeds len(tokenizer) ids and predicts output_classes of them; a file without `special` reads as
none, as step 9's do.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from indrajala_ml.data.text_data import PAD, SPECIAL_TOKENS, Vocabulary

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
    special: tuple[str, ...] = ()

    kind = "characters"

    def __post_init__(self) -> None:
        if self.special not in ((), SPECIAL_TOKENS):
            raise ValueError(f"special tokens are none or {list(SPECIAL_TOKENS)}; got {list(self.special)}")

    @classmethod
    def of(cls, corpus: str, text: str, special: tuple[str, ...] = ()) -> CharacterTokenizer:
        """The vocabulary of text, the corpus named corpus, with text's checksum, then special's tokens."""
        return cls(corpus, hashlib.sha256(text.encode()).hexdigest(), Vocabulary.of(text).symbols, special)

    @property
    def vocabulary(self) -> Vocabulary:
        return Vocabulary(self.symbols)

    def __len__(self) -> int:
        """Every id: the symbols', then the special tokens'."""
        return len(self.symbols) + len(self.special)

    @property
    def output_classes(self) -> int:
        """The ids a model predicts: all but the pad."""
        return len(self) - (PAD in self.special)

    def special_id(self, token: str) -> int:
        """A special token's id, after the symbols."""
        if token not in self.special:
            raise ValueError(f"the {self.corpus} tokenizer has no {token!r}")
        return len(self.symbols) + self.special.index(token)

    def encode(self, text: str) -> list[int]:
        """text's ids, refused if it holds a character the corpus doesn't."""
        unknown = sorted(set(text) - set(self.symbols))
        if unknown:
            raise ValueError(f"the {self.corpus} vocabulary has no {''.join(unknown)!r}")
        return self.vocabulary.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        """ids as text, a special token as its name."""
        count = len(self.symbols)
        return "".join(self.symbols[i] if i < count else self.special[i - count] for i in ids)

    def save(self, path: str | Path) -> None:
        state: dict[str, object] = {
            "kind": self.kind,
            "corpus": self.corpus,
            "sha256": self.sha256,
            "symbols": self.symbols,
        }
        if self.special:
            state["special"] = list(self.special)
        Path(path).write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("kind") not in KINDS:
            raise ValueError(f"{path}: an unknown tokenizer kind {state.get('kind')!r}; known: {', '.join(KINDS)}")
        return cls(state["corpus"], state["sha256"], state["symbols"], tuple(state.get("special", ())))
