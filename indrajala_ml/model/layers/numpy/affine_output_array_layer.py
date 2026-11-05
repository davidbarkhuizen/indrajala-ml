# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, which strict mode takes for constants)
"""
A linear output layer with a given loss (Dense(n, "linear", output=True, loss="given")): W x + b,
as AffineArrayLayer, whose output delta is the gradient the caller gives, the loss's gradient at
the output. learn_batch_output_gradient hands it in where a target would go, so the backward pass
and the optimizer are every network's. Array networks only (numpy and Rust).
"""

from __future__ import annotations

from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.residual_array_layer import AffineArrayLayer


class AffineOutputArrayLayer(AffineArrayLayer):
    """W x + b as the output layer: its delta is the given gradient at the output."""

    def compute_output_delta(self, reference: FloatArray, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.delta = reference

    def compute_output_delta_batch(self, reference_batch: FloatArray, *, context: ForwardContext = NO_CONTEXT) -> None:
        self.delta_batch = reference_batch
