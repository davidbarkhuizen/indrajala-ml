"""
A pure-Python batch's segments (the segments workplan, D4-D6; the cross-attention workplan, D13's
separator_opens): numpy's segment_context
(layers/numpy/segments.py) in Python lists, a row per example, so the pure-Python networks import
no numpy. The same rules: a separator ends its segment, a pad's segment is 0, positions count from
each segment's first token, and a target counts unless its input is a pad (the end-of-paragraph
token's counts: the context-length workplan, D11); a batch with neither id is NO_CONTEXT.
"""

from __future__ import annotations

from collections.abc import Sequence

from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext


def segment_rows(
    states: Sequence[Sequence[float]], separator: int, pad: int, separator_opens: bool = False
) -> ForwardContext:
    # with separator_opens (an encoder-decoder's target, the cross-attention workplan, D13), a
    # separator opens the segment after it: a token's segment is the separators up to and including
    # it, at least 1, numpy's rule
    rows = [[int(x) for x in state] for state in states]
    if not any(i in (separator, pad) for row in rows for i in row):
        return NO_CONTEXT
    segments: list[list[int]] = []
    positions: list[list[int]] = []
    counted: list[list[float]] = []
    for row in rows:
        segment_row: list[int] = []
        position_row: list[int] = []
        separators = 0
        for t, i in enumerate(row):
            if separator_opens:
                separators += i == separator
                segment = 0 if i == pad else max(separators, 1)
            else:
                segment = 0 if i == pad else 1 + separators
                separators += i == separator
            starts = t == 0 or segment != segment_row[-1]
            position_row.append(0 if starts else position_row[-1] + 1)
            segment_row.append(segment)
        segments.append(segment_row)
        positions.append(position_row)
        counted.append([0.0 if i == pad else 1.0 for i in row])
    count = int(sum(sum(row) for row in counted))
    return ForwardContext(segments, positions, counted, count, len(rows))


def blocked(segments: Sequence[int], query: int, key: int) -> bool:
    """Whether query may not attend to key by segment (D4): another segment, or a pad's, but never itself."""
    return query != key and (segments[key] != segments[query] or segments[key] == 0)
