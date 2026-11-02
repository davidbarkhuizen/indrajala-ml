from __future__ import annotations

from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.numpy import activations
from indrajala_ml.model.layers.numpy.array_layer import ArrayLayer, FloatArray
from indrajala_ml.model.layers.python.activations import Gelu


class GELUArrayLayer(ArrayLayer):
    """
    A GELU hidden layer over arrays (the FFN activations workplan, D1, D5): "gelu", z Phi(z) exactly
    through erf, or "gelu_tanh", GPT-2's approximation (activations.py), as GELULayer. Its
    derivative reads z, which the forward keeps; the update is ArrayLayer's. Hidden only:
    compute_output_delta* raise.
    """

    def __init__(self, size: int, input_size: int, activation: Gelu) -> None:
        super().__init__(size, input_size)
        self.activation: Gelu = activation

    def forward(self, x: FloatArray, *, context: ForwardContext = NO_CONTEXT) -> FloatArray:
        self.z = self.W @ x + self.b
        self.a = activations.activate(self.activation, self.z)
        return self.a

    def forward_batch(self, X: FloatArray, *, context: ForwardContext = NO_CONTEXT) -> FloatArray:
        self.Z = X @ self.W.T + self.b
        self.A = activations.activate(self.activation, self.Z)
        return self.A

    def compute_output_delta(self, reference: FloatArray, *, context: ForwardContext = NO_CONTEXT) -> None:
        raise NotImplementedError(
            "GELUArrayLayer is a hidden-layer activation, not an output one - an unbounded "
            "activation isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_output_delta_batch(self, reference_batch: FloatArray, *, context: ForwardContext = NO_CONTEXT) -> None:
        raise NotImplementedError(
            "GELUArrayLayer is a hidden-layer activation, not an output one - an unbounded "
            "activation isn't suited to any of this codebase's output-layer contracts."
        )

    def compute_hidden_delta(self, next_layer: ArrayLayer, *, context: ForwardContext = NO_CONTEXT) -> None:
        downstream = next_layer.downstream(context=context)
        self.delta = downstream * activations.derivative(self.activation, self.z)

    def compute_hidden_delta_batch(self, next_layer: ArrayLayer, *, context: ForwardContext = NO_CONTEXT) -> None:
        downstream = next_layer.downstream_batch(context=context)
        self.delta_batch = downstream * activations.derivative(self.activation, self.Z)
