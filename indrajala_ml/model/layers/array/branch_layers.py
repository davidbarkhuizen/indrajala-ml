# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, which strict mode takes for constants)
"""
A window's branches as layers (the window branches workplan, D5, D6), for either backend: they
only keep, pass on and add arrays. The Split keeps its input, and each branch starts from it; a
Branch ends a branch but the last, its input what a factor's pooling reads, and gives the next
branch the Split's input. The builder (array_layer_builder.py) wires each Split to its branches'
first layers and each Branch to its Split.

The backward pass keeps the network's loop, each layer reading the next one's downstream: the
layer before a Branch reads its poolings' gradient alone (the network's window backward), and the
Split's downstream is its first branch's first layer's plus each later branch's, in order, as the
crate's window program sums them. The backward pass runs in reverse, so every branch's deltas are
computed before the Split reads them; like the Rust fork, the Split sums them only when asked.
"""

from __future__ import annotations

from typing import Any

from indrajala_ml.model.specs.hidden_layers import Hidden, ParameterFree


class SplitArrayLayer[A](Hidden[A], ParameterFree[A]):
    """Branches' first layer: forward passes its input on and keeps it; its downstream sums its branches'."""

    # set by the builder: each branch's first layer, in order
    starts: list[Any]

    def __init__(self, size: int) -> None:
        self.size = size
        self.input_size = size
        self.starts = []

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
    A branch's end but the last: forward gives the next branch its Split's input. What reads the
    branch's result is a factor's pooling alone, so it has no delta and no downstream.
    """

    def __init__(self, split: SplitArrayLayer[A]) -> None:
        self.split = split
        self.size = split.size
        self.input_size = split.size

    def forward(self, x: A) -> A:
        return self.split.x

    def forward_batch(self, X: A) -> A:
        return self.split.X

    def compute_hidden_delta(self, next_layer: Any) -> None:
        pass

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        pass

    def downstream(self) -> A:
        raise AssertionError("a branch's end is read by its poolings alone (the window branches workplan)")

    def downstream_batch(self) -> A:
        raise AssertionError("a branch's end is read by its poolings alone (the window branches workplan)")
