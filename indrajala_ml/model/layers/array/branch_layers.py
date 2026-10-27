# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, which strict mode takes for constants)
"""
A window's branches as layers (the window branches workplan, D5, D6), for either backend: they
only keep, pass on and add arrays. The Split keeps its input, and each branch starts from it; a
Branch ends a branch but the last, its input what a factor's pooling reads, and gives the next
branch the Split's input. A Merge (the post-paper workplan's recombining window, D7) ends the last
branch of a merged split: its result is every branch's end summed in branch order, its own input
(the last branch's end) last, as the crate's window program's merge sums them. The builder
(array_layer_builder.py) wires each Split to its branches' first layers, each Branch to its Split
and each Merge to its Split.

The backward pass keeps the network's loop, each layer reading the next one's downstream: the
layer before a Branch reads its poolings' gradient alone (the network's window backward), and the
Split's downstream is its first branch's first layer's plus each later branch's, in order, as the
crate's window program sums them. The backward pass runs in reverse, so every branch's deltas are
computed before the Split reads them; like the Rust fork, the Split sums them only when asked.
A merged branch's end is read by its poolings and its Merge: the layer before its Branch reads the
Merge's delta plus its poolings' gradient (the network's window backward), and a Merge's delta is
its downstream, as an add's.
"""

from __future__ import annotations

from typing import Any

from indrajala_ml.model.specs.hidden_layers import DeltaIsDownstream, Hidden, ParameterFree


class SplitArrayLayer[A](Hidden[A], ParameterFree[A]):
    """Branches' first layer: forward passes its input on and keeps it; its downstream sums its branches'."""

    # set by the builder: each branch's first layer, in order; each Branch registers itself
    starts: list[Any]
    ends: list[BranchArrayLayer[A]]

    def __init__(self, size: int) -> None:
        self.size = size
        self.input_size = size
        self.starts = []
        self.ends = []

    def forward(self, x: A) -> A:
        self.x = x
        return x

    def forward_batch(self, X: A) -> A:
        self.X = X
        return X

    def compute_hidden_delta(self, next_layer: Any) -> None:
        # lazy: downstream() sums the branches when the layer before asks for it
        pass

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        pass

    def downstream(self) -> A:
        first, *later = self.starts
        total = first.downstream()
        for start in later:
            total = total + start.downstream()
        return total

    def downstream_batch(self) -> A:
        first, *later = self.starts
        total = first.downstream_batch()
        for start in later:
            total = total + start.downstream_batch()
        return total


class BranchArrayLayer[A](Hidden[A], ParameterFree[A]):
    """
    A branch's end but the last: forward keeps its input, the branch's result, and gives the next
    branch its Split's input. What reads the branch's result is its poolings and, if its split is
    merged, its Merge, so it has no delta and no downstream.
    """

    # set by the Merge that closes its split, if merged; the branch's result, kept for it
    merge: MergeArrayLayer[A] | None
    x: A
    X: A

    def __init__(self, split: SplitArrayLayer[A]) -> None:
        self.split = split
        self.size = split.size
        self.input_size = split.size
        self.merge = None
        split.ends.append(self)

    def forward(self, x: A) -> A:
        self.x = x
        return self.split.x

    def forward_batch(self, X: A) -> A:
        self.X = X
        return self.split.X

    def compute_hidden_delta(self, next_layer: Any) -> None:
        pass

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        pass

    def downstream(self) -> A:
        raise AssertionError("a branch's end is read by its poolings and its merge (the window branches workplan)")

    def downstream_batch(self) -> A:
        raise AssertionError("a branch's end is read by its poolings and its merge (the window branches workplan)")


class MergeArrayLayer[A](Hidden[A], DeltaIsDownstream[A], ParameterFree[A]):
    """
    A merged split's last layer (the post-paper workplan, D7): forward gives every branch's end
    summed in branch order, its own input (the last branch's end) last; its delta is the next
    layer's downstream, and its downstream its delta, as an add's.
    """

    def __init__(self, split: SplitArrayLayer[A]) -> None:
        self.split = split
        self.size = split.size
        self.input_size = split.size
        for end in split.ends:
            end.merge = self

    def forward(self, x: A) -> A:
        first, *later = self.split.ends
        total: Any = first.x
        for end in later:
            total = total + end.x
        return total + x

    def forward_batch(self, X: A) -> A:
        first, *later = self.split.ends
        total: Any = first.X
        for end in later:
            total = total + end.X
        return total + X

    def downstream(self) -> A:
        return self.delta

    def downstream_batch(self) -> A:
        return self.delta_batch
