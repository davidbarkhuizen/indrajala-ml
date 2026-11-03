"""
Incremental decoding of a sequence model (the generation workplan, D2, D3; the segments workplan,
D8): a Decoder holds one cache per token layer (decoding_protocols.py) and runs a batch of prompts,
then one token a row at a time, returning each time every row's last token's probabilities over
the vocabulary. A single prompt is a batch of one: prefill(ids) and step(id).

- prefill_batch(prompts) starts a new decode: the prompts left-padded to the longest (D8), their
  first T columns (the window) through every layer's prefill, then the rest one step each.
- step_batch(tokens) appends one token a row and runs them through every layer's step: one token's
  pass a row, each attention reading its cached keys and values.

Every row decodes at the same column, so a step's column (a rotary attention's angles) is one
number. A batch with segments (a model whose Embedding has them, D2, and a pad or a separator among
its tokens so far) carries each token's segment and position in every pass's context, by
segments.py's rules: a pad's segment is 0, a token's is 1 + the separators before it in its row,
and its position counts from its segment's first token (D5); a pad's position is 0 (its output is
discarded and its keys masked, D4). A batch without them carries none, and decodes as before
segments, bit for bit. A model without segments has no pad, so it decodes prompts of one length
only.

Past T tokens each attention's cache drops its oldest key and value; a rotary model's window then
slides exactly (its scores depend on distances up to T - 1, as in training), and so does a model
with no positions. A model with a learned Position has no row for position T: a row's cached decode
is exact while its current segment lies in the window, and once any row's position reaches T each
step is a full pass over every row's last T tokens, exact but uncached, for the rest of the decode,
and the Decoder says so once (D3 (a); the segments workplan, D8).

Implementation-free: the network gives its layers, its window, its special ids (or None) and four
hooks: the ids as its first layer's rows, a result's rows as lists, a full pass over rows of ids
(each row's last token's probabilities), and a decode pass's context from its segments and
positions (lists of rows, or None), its N and its column.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from indrajala_ml.model.layers.forward_context import ForwardContext
from indrajala_ml.model.protocols.decoding_protocols import DecodingLayer, refuse_undecodable

Probabilities = list[float]
Rows = list[list[int]]


@dataclass(frozen=True)
class SpecialIds:
    """
    A model's separator and pad ids (the segments workplan, D2), and whether a separator opens the
    segment after it (an encoder-decoder's target, the cross-attention workplan, D13, D16) rather
    than ending its own.
    """

    separator: int
    pad: int
    separator_opens: bool = False


@dataclass
class _Row:
    # one row's decode so far: its ids, pads included, and its last token's segment and position,
    # whether that token was a separator, and the separators so far (D16's rule)
    ids: list[int]
    segment: int = 0
    position: int = 0
    separated: bool = False
    separators: int = 0


class Decoder:
    """One decode of a network: prefill_batch(prompts) (or prefill(ids)), then step_batch(tokens) (step(id))."""

    def __init__(
        self,
        layers: Sequence[Any],
        tokens: int,
        ids_input: Callable[[Sequence[int]], Any],
        rows: Callable[[Any], list[list[float]]],
        full_pass: Callable[[Rows], list[Probabilities]],
        context: Callable[[Rows | None, Rows | None, int, int], ForwardContext],
        learned_positions: bool,
        special: SpecialIds | None = None,
    ) -> None:
        refuse_undecodable(list(layers))
        self.layers: list[DecodingLayer[Any]] = list(layers)
        self.tokens = tokens
        self._ids_input = ids_input
        self._rows = rows
        self._full_pass = full_pass
        self._context = context
        self.learned_positions = learned_positions
        self.special = special
        # refuses a layer that can't decode (a non-causal attention) at once
        self._caches: list[Any] = [layer.start_decoding() for layer in self.layers]
        self._batch: list[_Row] = []
        self._segmented = False
        self._uncached_from_here = False
        self._warned = False

    @property
    def position(self) -> int:
        """The next token's column: the columns decoded so far, pads included."""
        return len(self._batch[0].ids) if self._batch else 0

    def prefill(self, ids: Sequence[int]) -> Probabilities:
        """A new decode of one prompt; its last token's probabilities."""
        return self.prefill_batch([ids])[0]

    def step(self, token: int) -> Probabilities:
        """Appends one token to a batch of one; its probabilities for the next."""
        return self.step_batch([token])[0]

    def prefill_batch(self, prompts: Sequence[Sequence[int]]) -> list[Probabilities]:
        """A new decode of the prompts, left-padded to the longest; each row's last token's probabilities."""
        assert prompts and all(len(p) >= 1 for p in prompts), "a decode has at least one prompt of at least one token"
        L = max(len(p) for p in prompts)
        if self.special is None and any(len(p) != L for p in prompts):
            raise ValueError(
                "a model without segments has no pad (the segments workplan, D2, D8): its prompts in one batch have "
                f"one length; got lengths {sorted({len(p) for p in prompts})}"
            )
        pad = None if self.special is None else self.special.pad
        assert pad is None or all(pad not in p for p in prompts), "a prompt holds no pad: the Decoder pads it"
        padded = [[pad] * (L - len(p)) + list(p) for p in prompts] if pad is not None else [list(p) for p in prompts]
        self._caches = [layer.start_decoding() for layer in self.layers]
        self._batch = [_Row([]) for _ in padded]
        self._segmented = False
        self._uncached_from_here = False
        head = min(L, self.tokens)
        segments, positions = self._advance([row[:head] for row in padded])
        out = self._ids_input([i for row in padded for i in row[:head]])
        context = self._context(segments, positions, len(padded), 0)
        for layer, cache in zip(self.layers, self._caches, strict=True):
            out = layer.prefill(out, cache, context)
        probabilities = self._rows(out)[head - 1 :: head]
        for column in range(head, L):
            probabilities = self.step_batch([row[column] for row in padded])
        return probabilities

    def step_batch(self, tokens: Sequence[int]) -> list[Probabilities]:
        """Appends one token a row; each row's probabilities for the next."""
        assert self._batch, "a decode starts with prefill"
        assert len(tokens) == len(self._batch), f"a step takes one token a row, {len(self._batch)}; got {len(tokens)}"
        column = self.position
        segments, positions = self._advance([[token] for token in tokens])
        if self.learned_positions and (
            self._uncached_from_here or any(row.segment != 0 and row.position >= self.tokens for row in self._batch)
        ):
            self._uncached_from_here = True
            return self._uncached()
        out = self._ids_input(list(tokens))
        context = self._context(segments, positions, len(tokens), column)
        for layer, cache in zip(self.layers, self._caches, strict=True):
            out = layer.step(out, cache, context)
        return self._rows(out)

    def _advance(self, new: Rows) -> tuple[Rows | None, Rows | None]:
        # appends each row's new ids, tracking segments and positions; the new tokens' segments and
        # positions, or None for both while the batch has no special id (D9's path)
        special = self.special
        segments: Rows = []
        positions: Rows = []
        for row, ids in zip(self._batch, new, strict=True):
            row_segments: list[int] = []
            row_positions: list[int] = []
            for i in ids:
                if special is not None and i == special.pad:
                    row.segment, row.position = 0, 0
                elif special is not None and special.separator_opens:
                    # a separator opens its segment: the separators up to and including the token,
                    # at least 1, segment_context(..., separator_opens=True)'s rule (D16)
                    row.separators += i == special.separator
                    segment = max(row.separators, 1)
                    row.position = 0 if segment != row.segment else row.position + 1
                    row.segment = segment
                elif row.segment == 0 or row.separated:
                    row.segment, row.position = row.segment + 1, 0
                else:
                    row.position += 1
                row.separated = special is not None and i == special.separator
                self._segmented = self._segmented or (special is not None and i in (special.separator, special.pad))
                row.ids.append(i)
                row_segments.append(row.segment)
                row_positions.append(row.position)
            segments.append(row_segments)
            positions.append(row_positions)
        if not self._segmented:
            return None, None
        return segments, positions

    def _uncached(self) -> list[Probabilities]:
        if not self._warned:
            warnings.warn(
                f"a learned Position has no row past the window's {self.tokens} tokens: each token from here is a "
                "full pass over the last tokens, uncached (the generation workplan, D3)",
                stacklevel=3,
            )
            self._warned = True
        return self._full_pass([row.ids[-self.tokens :] for row in self._batch])
