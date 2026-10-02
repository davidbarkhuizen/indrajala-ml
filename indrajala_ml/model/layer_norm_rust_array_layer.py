# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, which strict mode takes for constants)
"""
Layer norm on the Rust backend (the layer-norm and attention workplan, D5 and D9): each method one
call to the crate's layer_norm.rs, which computes the README's expressions (Layer norm and
attention) as LayerNormArrayLayer does, so given the same inputs it computes the same bits. A
single example is one call too, on the 1D vector, with a batch of one's bits.
"""

from __future__ import annotations

from typing import Any

import indrajala_math_rust as pa

from indrajala_ml.model.array_parameters import GammaAndBeta


class LayerNormRustArrayLayer(GammaAndBeta[pa.Array]):
    """
    LayerNormArrayLayer on the Rust backend: y = gamma * xhat + beta over each of tokens tokens of
    features features (a flat layer is one token), gamma and beta (features,), starting at 1 and 0
    and never decayed. Hidden only.

    It has no W, so a sigmoid, ReLU or dropout layer before it takes its downstream and a mask op
    rather than the fused hidden delta (D5; rust_array_layer.reads_next_w).
    """

    def __init__(self, tokens: int, features: int, epsilon: float) -> None:
        self.tokens = tokens
        self.features = features
        self.size = tokens * features
        self.input_size = self.size
        self.epsilon = epsilon

        # nothing drawn
        self.gamma = pa.Array([1.0] * features)
        self.beta = pa.Array.zeros(features)
        self.reset_gradient_accum()

    def forward(self, x: pa.Array) -> pa.Array:
        self.a, self._xhat, self._std = pa.layer_norm_forward(x, self.gamma, self.beta, self.epsilon)
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self.A, self._xhat, self._std = pa.layer_norm_forward_batch(X, self.gamma, self.beta, self.epsilon)
        return self.A

    def compute_output_delta(self, reference: pa.Array) -> None:
        raise NotImplementedError("a layer-norm layer is hidden")

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        raise NotImplementedError("a layer-norm layer is hidden")

    def compute_hidden_delta(self, next_layer: Any) -> None:
        self.delta = next_layer.downstream()

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        self.delta_batch = next_layer.downstream_batch()

    def downstream(self) -> pa.Array:
        return pa.layer_norm_downstream_batch(self.delta, self.gamma, self._xhat, self._std)

    def downstream_batch(self) -> pa.Array:
        return pa.layer_norm_downstream_batch(self.delta_batch, self.gamma, self._xhat, self._std)

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        self.grad_gamma, self.grad_beta = pa.layer_norm_accumulate_gradient_batch(
            self.delta, self._xhat, self.grad_gamma, self.grad_beta
        )

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        self.grad_gamma, self.grad_beta = pa.layer_norm_accumulate_gradient_batch(
            self.delta_batch, self._xhat, self.grad_gamma, self.grad_beta
        )

    def reset_gradient_accum(self) -> None:
        self.grad_gamma = pa.Array.zeros(self.features)
        self.grad_beta = pa.Array.zeros(self.features)
