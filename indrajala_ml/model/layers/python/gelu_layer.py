from __future__ import annotations

from typing import ClassVar

from indrajala_ml.model.layers.python import activations
from indrajala_ml.model.layers.python.activations import Gelu
from indrajala_ml.model.layers.python.backprop_layer import BackpropLayer
from indrajala_ml.model.layers.python.backprop_node import BackpropNode


class GELUNode(BackpropNode):
    """
    A hidden node with GELU (the FFN activations workplan, D1, D5), z Phi(z) exactly through erf
    (activations.py); GELUTanhNode is GPT-2's approximation. Its derivative reads z, which forward
    keeps per example (example_fields). Hidden only, as ReLUNode.
    """

    activation: ClassVar[Gelu] = "gelu"
    example_fields: ClassVar[tuple[str, ...]] = ("_activation", "_z", "delta")

    def forward(self) -> float:
        self._z = self.z()
        self._activation = activations.activate(self.activation, self._z)
        return self._activation

    def compute_output_delta(self, reference_value: float) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is a hidden-layer activation, not an output one - an unbounded activation "
            "isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_hidden_delta(self, downstream: float) -> None:
        self.delta = downstream * activations.derivative(self.activation, self._z)


class GELUTanhNode(GELUNode):
    """GELUNode with GPT-2's tanh approximation."""

    activation: ClassVar[Gelu] = "gelu_tanh"


class GELULayer(BackpropLayer):
    """A hidden layer of GELUNodes."""

    _node_cls = GELUNode


class GELUTanhLayer(BackpropLayer):
    """A hidden layer of GELUTanhNodes."""

    _node_cls = GELUTanhNode
