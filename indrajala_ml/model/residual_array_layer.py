# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, Z, which strict mode takes for constants)
"""
A residual block's layers in numpy (the residual-connections workplan; README, Residual
connections): the block's Fork, which keeps its input, the Add, which adds it to the body's output,
and the affine layer that ends a body. The builder (array_layer_builder.py) wires each fork to its
add and to its body's first layer.

The network's loops don't change: each layer reads only the next one in the backward pass, and the
fork reaches its add through the reference it is built with. The backward pass runs in reverse, so
the add's delta and the body's are computed before the fork reads them.
"""

from __future__ import annotations

from typing import Any

from indrajala_ml.model.array_layer import ArrayLayer, FloatArray
from indrajala_ml.model.hidden_layers import DeltaIsDownstream, Hidden, ParameterFree


class AffineArrayLayer(Hidden[FloatArray], DeltaIsDownstream[FloatArray], ArrayLayer):
    """
    W x + b with no activation (the residual-connections workplan, D4): a residual block's body's
    last layer. Its delta is its downstream from the add after it, since the identity's derivative
    is 1; its downstream and gradients are ArrayLayer's. Hidden only.
    """

    def forward(self, x: FloatArray) -> FloatArray:
        self.a = self.W @ x + self.b
        return self.a

    def forward_batch(self, X: FloatArray) -> FloatArray:
        self.A = X @ self.W.T + self.b
        return self.A


class ForkArrayLayer(Hidden[FloatArray], ParameterFree[FloatArray]):
    """
    A residual block's first layer: forward passes its input on unchanged, and keeps it (x, or X
    for a batch) for its add. Its delta is the sum of the two paths' gradients, the body's
    (body_first's downstream) and the identity's (the add's delta), and is its downstream.
    """

    # set by the builder: the block's add, and its body's first layer
    add: AddArrayLayer
    body_first: Any

    def __init__(self, size: int) -> None:
        self.size = size
        self.input_size = size

    def forward(self, x: FloatArray) -> FloatArray:
        self.x = x
        return x

    def forward_batch(self, X: FloatArray) -> FloatArray:
        self.X = X
        return X

    def compute_hidden_delta(self, next_layer: Any) -> None:
        # next_layer is body_first
        self.delta = next_layer.downstream() + self.add.delta

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        self.delta_batch = next_layer.downstream_batch() + self.add.delta_batch

    def downstream(self) -> FloatArray:
        return self.delta

    def downstream_batch(self) -> FloatArray:
        return self.delta_batch


class AddArrayLayer(Hidden[FloatArray], DeltaIsDownstream[FloatArray], ParameterFree[FloatArray]):
    """
    A residual block's last layer: the body's output plus the block's input, y + x, which its fork
    kept. Its delta is its downstream from the next layer, and its downstream is its delta, which
    the body's affine layer and the fork both read: the add's derivative is 1 on both paths.
    """

    def __init__(self, fork: ForkArrayLayer) -> None:
        self.fork = fork
        self.size = fork.size
        self.input_size = fork.size

    def forward(self, x: FloatArray) -> FloatArray:
        self.a = x + self.fork.x
        return self.a

    def forward_batch(self, X: FloatArray) -> FloatArray:
        self.A = X + self.fork.X
        return self.A

    def downstream(self) -> FloatArray:
        return self.delta

    def downstream_batch(self) -> FloatArray:
        return self.delta_batch
