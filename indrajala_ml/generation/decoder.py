"""
Incremental decoding of a sequence model (the generation workplan, D2, D3): a Decoder holds one
cache per token layer (decoding_protocols.py) and runs the prompt, then one token at a time,
returning each time the last token's probabilities over the vocabulary.

- prefill(ids) starts a new decode: the prompt's first T tokens (the window) through every layer's
  prefill, then the rest one step each.
- step(id) appends one token and runs it through every layer's step: one token's pass, each
  attention reading its cached keys and values.

Past T tokens each attention's cache drops its oldest key and value; a rotary model's window then
slides exactly (its scores depend on distances up to T - 1, as in training), and so does a model
with no positions. A model with a learned Position has no row for position T: from there each token
is a full pass over the last T tokens, exact but uncached, and the Decoder says so once (D3 (a)).

Implementation-free: the network gives its layers, its window and three hooks, the ids as its first
layer's input, a result's last row as a list, and its full pass's last token.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

from indrajala_ml.model.protocols.decoding_protocols import DecodingLayer, refuse_undecodable

Probabilities = list[float]


class Decoder:
    """One decode of a network: prefill(ids), then step(id) per token."""

    def __init__(
        self,
        layers: Sequence[Any],
        tokens: int,
        ids_input: Callable[[Sequence[int]], Any],
        last_row: Callable[[Any], Probabilities],
        full_pass: Callable[[Sequence[int]], Probabilities],
        learned_positions: bool,
    ) -> None:
        refuse_undecodable(list(layers))
        self.layers: list[DecodingLayer[Any]] = list(layers)
        self.tokens = tokens
        self._ids_input = ids_input
        self._last_row = last_row
        self._full_pass = full_pass
        self.learned_positions = learned_positions
        # refuses a layer that can't decode (a non-causal attention) at once
        self._caches: list[Any] = [layer.start_decoding() for layer in self.layers]
        self._history: list[int] = []
        self._warned = False

    @property
    def position(self) -> int:
        """The next token's position: the tokens decoded so far."""
        return len(self._history)

    def prefill(self, ids: Sequence[int]) -> Probabilities:
        """A new decode of the prompt ids; the last token's probabilities."""
        assert len(ids) >= 1, "a prompt has at least one token"
        self._caches = [layer.start_decoding() for layer in self.layers]
        self._history = []
        head = list(ids[: self.tokens])
        out = self._ids_input(head)
        for layer, cache in zip(self.layers, self._caches, strict=True):
            out = layer.prefill(out, cache)
        self._history = head
        probabilities = self._last_row(out)
        for token in ids[self.tokens :]:
            probabilities = self.step(token)
        return probabilities

    def step(self, token: int) -> Probabilities:
        """Appends one token; its probabilities for the next."""
        assert self._history, "a decode starts with prefill"
        position = self.position
        self._history.append(token)
        if self.learned_positions and position >= self.tokens:
            return self._uncached()
        out = self._ids_input([token])
        for layer, cache in zip(self.layers, self._caches, strict=True):
            out = layer.step(out, cache, position)
        return self._last_row(out)

    def _uncached(self) -> Probabilities:
        if not self._warned:
            warnings.warn(
                f"a learned Position has no row past the window's {self.tokens} tokens: each token from here is a "
                "full pass over the last tokens, uncached (the generation workplan, D3)",
                stacklevel=3,
            )
            self._warned = True
        return self._full_pass(self._history[-self.tokens :])
