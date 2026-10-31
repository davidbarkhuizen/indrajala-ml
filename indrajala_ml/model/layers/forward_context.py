"""
The forward context (the segments workplan, D3): what a batch tells every layer beyond its input,
built once per batch by the network and passed to every layer's forward, backward and gradient
passes, so what a layer reads is in its signature and nothing is left over between batches.

Empty for now (stage 2: every pass takes it, nothing reads it). The segments workplan's stage 4
adds the batch's segment ids (the mask in attention, positions per segment, the loss's counted
tokens); later steps add step 9's decode positions and step 15's encoder output. A layer that
reads nothing from it ignores it. Backend-free: the pure-Python layers take it too.

Every pass's context is keyword-only, defaulting to NO_CONTEXT, so a layer driven alone (a test, a
benchmark) needs none; the networks always pass theirs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardContext:
    """One batch's information for its layers; none yet."""


NO_CONTEXT = ForwardContext()
