# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, U, X, H, which strict mode takes for constants)
"""
A window-recurrent network's layers on the Rust backend (the window-recurrent adder workplan,
stage 5): recurrent_array_layer.py's StateInputArrayLayer, AttentionPoolingArrayLayer,
GRUArrayLayer, CodesArrayLayer and InitialStateArrayLayer, with the same methods, each pass one
call of the crate's recurrent.rs. Their products are the crate's, so they match numpy's to within
rounding; every other step, and the codes' ops entirely, are numpy's by bits. Each draws what its
numpy twin draws, in the same order, from the crate's Generator (numpy's default_rng), so the same
seed gives the same weights by bits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import indrajala_math_rust as pa

from indrajala_ml.model.layers.rust.rust_array_layer import (
    fan_in_aware_random_rust_layer,
    fan_in_aware_random_rust_weights,
)


class StateInputRustArrayLayer:
    """StateInputArrayLayer on the Rust backend (state_input_forward, state_input_backward)."""

    decayed: ClassVar[tuple[bool, ...]] = (True,)

    def __init__(self, tokens: int, features: int, state_size: int) -> None:
        self.tokens = tokens
        self.features = features
        self.state_size = state_size
        self.size = tokens * features
        self.input_size = self.size
        self.P = pa.Array.zeros((features, state_size))
        self.grad_P = pa.Array.zeros((features, state_size))
        self.state = pa.Array.zeros((1, state_size))

    def parameters(self) -> tuple[pa.Array, ...]:
        return (self.P,)

    def gradients(self) -> tuple[pa.Array, ...]:
        return (self.grad_P,)

    def set_parameters(self, parameters: Sequence[pa.Array]) -> None:
        (self.P,) = parameters

    def reset_gradient_accum(self) -> None:
        self.grad_P = pa.Array.zeros((self.features, self.state_size))

    def draw(self, rng: pa.Generator) -> None:
        self.P = fan_in_aware_random_rust_weights(rng, self.features, self.state_size)

    def set_state(self, state: pa.Array) -> None:
        """The batch's states, (N, state_size), which the next forward pass adds."""
        self.state = state

    def forward_batch(self, X: pa.Array) -> pa.Array:
        return pa.state_input_forward(X, self.state, self.P, self.tokens)

    def forward(self, x: pa.Array) -> pa.Array:
        return self.forward_batch(x.reshape((1, self.size))).reshape(self.size)

    def compute_output_delta(self, reference: pa.Array) -> None:
        raise NotImplementedError("a StateInputRustArrayLayer is hidden, inside a network")

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        raise NotImplementedError("a StateInputRustArrayLayer is hidden, inside a network")

    def compute_hidden_delta(self, next_layer: Any) -> None:
        self._backward(next_layer.downstream().reshape((1, self.size)))

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        self._backward(next_layer.downstream_batch())

    def _backward(self, downstream: pa.Array) -> None:
        # the state's gradient now; P's, which the backward op gives with it, kept for
        # accumulate_gradient_batch
        self.delta_batch = downstream
        self.state_delta, self._grad_P = pa.state_input_backward(
            downstream, self.state, self.P, self.grad_P, self.tokens
        )

    def downstream(self) -> pa.Array:
        return self.delta_batch.reshape(self.size)

    def downstream_batch(self) -> pa.Array:
        return self.delta_batch

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        self.grad_P = self._grad_P

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        self.grad_P = self._grad_P


class AttentionPoolingRustArrayLayer:
    """AttentionPoolingArrayLayer on the Rust backend (attention_pooling_forward, attention_pooling_backward)."""

    decayed: ClassVar[tuple[bool, ...]] = (False,)

    def __init__(self, tokens: int, features: int) -> None:
        self.tokens = tokens
        self.features = features
        self.q = pa.Array.zeros(features)
        self.grad_q = pa.Array.zeros(features)

    def parameters(self) -> tuple[pa.Array, ...]:
        return (self.q,)

    def gradients(self) -> tuple[pa.Array, ...]:
        return (self.grad_q,)

    def set_parameters(self, parameters: Sequence[pa.Array]) -> None:
        (self.q,) = parameters

    def reset_gradient_accum(self) -> None:
        self.grad_q = pa.Array.zeros(self.features)

    def forward_batch(self, X: pa.Array) -> pa.Array:
        """X (N, T * d), the tokens, to u (N, d)."""
        self._X = X
        u, self._p = pa.attention_pooling_forward(X, self.q, self.tokens)
        return u

    def backward_batch(self, dU: pa.Array) -> pa.Array:
        """u's gradient (N, d) to the tokens' (N, T * d); q's gradient accumulated."""
        dX, self.grad_q = pa.attention_pooling_backward(self._X, self._p, self.q, dU, self.grad_q, self.tokens)
        return dX


class GRURustArrayLayer:
    """
    GRUArrayLayer on the Rust backend (gru_forward, gru_backward). Each gate's [W U] and b are
    drawn as GRUArrayLayer draws them, then split into W's and U's columns and stacked by gate.
    """

    decayed: ClassVar[tuple[bool, ...]] = (True, True, False)

    def __init__(self, input_size: int, state_size: int, update_bias: float = 1.0) -> None:
        self.input_size = input_size
        self.state_size = state_size
        self.update_bias = update_bias
        rows = 3 * state_size
        self.W = pa.Array.zeros((rows, input_size))
        self.U = pa.Array.zeros((rows, state_size))
        self.b = pa.Array.zeros(rows)
        self.reset_gradient_accum()

    def parameters(self) -> tuple[pa.Array, ...]:
        return self.W, self.U, self.b

    def gradients(self) -> tuple[pa.Array, ...]:
        return self.grad_W, self.grad_U, self.grad_b

    def set_parameters(self, parameters: Sequence[pa.Array]) -> None:
        self.W, self.U, self.b = parameters

    def reset_gradient_accum(self) -> None:
        rows = 3 * self.state_size
        self.grad_W = pa.Array.zeros((rows, self.input_size))
        self.grad_U = pa.Array.zeros((rows, self.state_size))
        self.grad_b = pa.Array.zeros(rows)

    def draw(self, rng: pa.Generator) -> None:
        s, d = self.state_size, self.input_size
        gates = [fan_in_aware_random_rust_layer(rng, s, d + s) for _ in range(3)]
        rows = [row for W, _ in gates for row in W.tolist()]
        self.W = pa.Array([row[:d] for row in rows])
        self.U = pa.Array([row[d:] for row in rows])
        biases = [value for _, b in gates for value in b.tolist()]
        biases[s : 2 * s] = [self.update_bias] * s
        self.b = pa.Array(biases)

    def forward_batch(self, U_in: pa.Array, C: pa.Array) -> pa.Array:
        """u (N, d) and c (N, s) to h (N, s)."""
        self._u, self._c = U_in, C
        H, *self._saved = pa.gru_forward(U_in, C, self.W, self.U, self.b)
        return H

    def backward_batch(self, dH: pa.Array) -> tuple[pa.Array, pa.Array]:
        """h's gradient (N, s) to u's (N, d) and c's (N, s); the gradients of W, U and b accumulated."""
        r, z, h_tilde, rc = self._saved
        dU_in, dC, self.grad_W, self.grad_U, self.grad_b = pa.gru_backward(
            dH, self._u, self._c, r, z, h_tilde, rc, self.W, self.U, self.grad_W, self.grad_U, self.grad_b
        )
        return dU_in, dC


class CodesRustArrayLayer:
    """CodesArrayLayer on the Rust backend (codes_forward, codes_backward): numpy's by bits."""

    decayed: ClassVar[tuple[bool, ...]] = (False,)

    def __init__(self, count: int, size: int, commitment: float = 0.25) -> None:
        self.count = count
        self.size = size
        self.commitment = commitment
        self.E = pa.Array.zeros((count, size))
        self.grad_E = pa.Array.zeros((count, size))

    def parameters(self) -> tuple[pa.Array, ...]:
        return (self.E,)

    def gradients(self) -> tuple[pa.Array, ...]:
        return (self.grad_E,)

    def set_parameters(self, parameters: Sequence[pa.Array]) -> None:
        (self.E,) = parameters

    def reset_gradient_accum(self) -> None:
        self.grad_E = pa.Array.zeros((self.count, self.size))

    def draw(self, rng: pa.Generator) -> None:
        self.E = fan_in_aware_random_rust_weights(rng, self.count, self.size)

    def nearest(self, H: pa.Array) -> list[int]:
        """Each row of H's nearest code's index."""
        return pa.codes_forward(H, self.E)[1]

    def forward_batch(self, H: pa.Array) -> pa.Array:
        """h (N, size) to its nearest code (N, size), a new array."""
        self._H = H
        self._Q, self.indices = pa.codes_forward(H, self.E)
        return self._Q

    def loss_batch(self, weight: float) -> float:
        """CodesArrayLayer.loss_batch: the squares a left fold over each row, then over the rows."""
        total = 0.0
        for h_row, q_row in zip(self._H.tolist(), self._Q.tolist(), strict=True):
            row = 0.0
            for h, q in zip(h_row, q_row, strict=True):
                row += (h - q) * (h - q)
            total += row
        return float(weight * (1.0 + self.commitment) * total)

    def backward_batch(self, dQ: pa.Array, weight: float) -> pa.Array:
        """The code's gradient (N, size) to h's: straight through, plus the commitment loss's; E's accumulated."""
        dH, self.grad_E = pa.codes_backward(
            dQ, self._H, self._Q, self.indices, self.grad_E, weight, self.commitment
        )
        return dH


class InitialStateRustArrayLayer:
    """InitialStateArrayLayer on the Rust backend: its gradient the snapped state's summed over the examples."""

    decayed: ClassVar[tuple[bool, ...]] = (False,)

    def __init__(self, state_size: int) -> None:
        self.state_size = state_size
        self.c = pa.Array.zeros(state_size)
        self.grad_c = pa.Array.zeros(state_size)

    def parameters(self) -> tuple[pa.Array, ...]:
        return (self.c,)

    def gradients(self) -> tuple[pa.Array, ...]:
        return (self.grad_c,)

    def set_parameters(self, parameters: Sequence[pa.Array]) -> None:
        (self.c,) = parameters

    def reset_gradient_accum(self) -> None:
        self.grad_c = pa.Array.zeros(self.state_size)

    def forward_batch(self, examples: int) -> pa.Array:
        """c for each of examples examples, (examples, state_size), a new array."""
        return pa.Array.zeros((examples, self.state_size)) + self.c

    def backward_batch(self, dC: pa.Array) -> None:
        self.grad_c = self.grad_c + pa.sum_axis0(dC)

