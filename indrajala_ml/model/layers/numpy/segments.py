"""
A numpy batch's segments (the segments workplan, D4-D6), from its token ids: each token's segment
id, its position within its segment, and whether its target counts.

- A separator ends its paragraph, so it belongs to it: a token's segment is 1 + the separators
  before it, and a pad's is 0. Pads stand before a row's first real token (left padding, D8).
- A position counts from its segment's first token (D5).
- A target counts unless its input is a pad (D6; the context-length workplan, D11): the separator is
  the end-of-paragraph token, so its target, the next paragraph's first character, counts,
  predicted from the paragraph the token ends.

A batch with neither a separator nor a pad has one segment a row and every target counted: its
context is NO_CONTEXT, so it trains by today's path, bit for bit (D9).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.numpy.array_layer import FloatArray


def segment_context(X: FloatArray, separator: int, pad: int) -> ForwardContext:
    """X's context: X is one example's (T,) ids or a batch's (N, T), as floats."""
    ids = np.atleast_2d(X).astype(np.intp)
    is_separator, is_pad = ids == separator, ids == pad
    if not (is_separator.any() or is_pad.any()):
        return NO_CONTEXT
    before = np.cumsum(is_separator, axis=1) - is_separator
    segments = np.where(is_pad, 0, 1 + before)
    tokens: npt.NDArray[np.intp] = np.arange(ids.shape[1], dtype=np.intp)
    starts = np.ones(ids.shape, dtype=bool)
    starts[:, 1:] = segments[:, 1:] != segments[:, :-1]
    first: npt.NDArray[np.intp] = np.maximum.accumulate(np.where(starts, tokens, 0), axis=1)
    positions = tokens - first
    counted = (~is_pad).astype(np.float64)
    return ForwardContext(
        segments=segments, positions=positions, counted=counted, count=int(counted.sum()), examples=ids.shape[0]
    )


def attention_blocked(segments: npt.NDArray[np.intp]) -> npt.NDArray[np.bool_]:
    """(N, T, T): whether query t may not attend to key u by segment (D4): another segment, or a pad's, but never itself."""
    blocked = (segments[:, :, None] != segments[:, None, :]) | (segments[:, None, :] == 0)
    T = segments.shape[1]
    blocked[:, np.arange(T), np.arange(T)] = False
    return blocked
