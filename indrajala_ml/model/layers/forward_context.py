"""
The forward context (the segments workplan, D3): what a batch tells every layer beyond its input,
built once per batch by the network and passed to every layer's forward, backward and gradient
passes, so what a layer reads is in its signature and nothing is left over between batches.

Its fields are the batch's segments (the segments workplan, D4-D6), all None when the batch has
none (no separator, no pad: D9's path, by bits as before): segments, each token's segment id (0 for
a pad, then 1, 2, ... a paragraph each); positions, each token's position from its segment's first
token; counted, whether each token's target counts in the loss (1.0) or not (0.0); and count, the
batch's counted targets. Each is an (N, T) array of the backend's, built by the network
(segments.py on numpy). Later steps add step 9's decode positions and step 15's encoder output. A
layer that reads nothing from it ignores it. Backend-free: the pure-Python layers take it too.

Every pass's context is keyword-only, defaulting to NO_CONTEXT, so a layer driven alone (a test, a
benchmark) needs none; the networks always pass theirs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, eq=False)
class ForwardContext:
    """One batch's information for its layers: its segments, or none (eq=False: arrays don't compare)."""

    segments: Any = None
    positions: Any = None
    counted: Any = None
    count: int = 0


NO_CONTEXT = ForwardContext()
