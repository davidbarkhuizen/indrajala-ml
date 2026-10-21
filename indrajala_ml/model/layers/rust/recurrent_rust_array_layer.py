# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, U, X, H, which strict mode takes for constants)
"""
A window-recurrent network's layers on the Rust backend (the window-recurrent adder workplan,
stage 5; the factored-state workplan, stage 4): recurrent_array_layer.py's StateInputArrayLayer,
each pass one call of the crate's recurrent.rs, and RecurrenceArrayLayer, a window's recurrence
over its state's factors, one call of the crate's factored.rs each way. The factors' parts hold
their parameters and gradients: the pooling's query, the cell's weights, the codes and the initial
state. The crate's products are its own, so they match numpy's to within rounding; every other
step, and the codes' entirely, are numpy's by bits. Each part draws what its numpy twin draws, in
the same order, from the crate's Generator (numpy's default_rng), so the same seed gives the same
weights by bits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import indrajala_math_rust as pa

from indrajala_ml.model.layers.rust.rust_array_layer import (
    fan_in_aware_random_rust_layer,
    fan_in_aware_random_rust_weights,
)
from indrajala_ml.model.specs.window_recurrence import (
    Consolidation,
    check_continuous,
    check_couplings,
    check_temperatures,
    commitment_targets,
    set_commitments,
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
    """AttentionPoolingArrayLayer's query on the Rust backend: RecurrenceRustArrayLayer runs it."""

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

class GRURustArrayLayer:
    """
    GRUArrayLayer's weights on the Rust backend: RecurrenceRustArrayLayer runs it. Each gate's [W U]
    and b are drawn as GRUArrayLayer draws them, then split into W's and U's columns and stacked by
    gate.
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

class CodesRustArrayLayer:
    """CodesArrayLayer's codes on the Rust backend: RecurrenceRustArrayLayer snaps to them, numpy's by bits."""

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

class InitialStateRustArrayLayer:
    """InitialStateArrayLayer's vector on the Rust backend: RecurrenceRustArrayLayer starts from it."""

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


class _FactorParts:
    """One factor's parts: its pooling, cell, codes (None: continuous) and initial state."""

    def __init__(
        self,
        pooling: AttentionPoolingRustArrayLayer,
        cell: GRURustArrayLayer,
        codes: CodesRustArrayLayer | None,
        initial_state: InitialStateRustArrayLayer,
    ) -> None:
        self.pooling = pooling
        self.cell = cell
        self.codes = codes
        self.initial_state = initial_state


class RecurrenceRustArrayLayer:
    """
    RecurrenceArrayLayer on the Rust backend: a window's every factor in one call each way
    (recurrence_step_forward, recurrence_step_backward), the initial state's codes in one
    (factored_codes_forward, factored_codes_backward), its factors' parts holding their parameters
    and gradients. The crate's ops are the single-factor ops factor by factor, so with one factor
    this is the window's step as it was before factors, by bits.
    """

    def __init__(
        self, sizes: Sequence[tuple[int, int | None, float]], tokens: int, features: int, cross: bool, update_bias: float
    ) -> None:
        """sizes: each factor's (state size, code count or None, commitment)."""
        self.sizes = [s for s, _, _ in sizes]
        self.tokens = tokens
        self.cross = cross
        self.state_size = sum(self.sizes)
        self.starts = [sum(self.sizes[:g]) for g in range(len(self.sizes))]
        self.factors = [
            _FactorParts(
                AttentionPoolingRustArrayLayer(tokens, features),
                GRURustArrayLayer(features + (self.state_size - s if cross else 0), s, update_bias),
                None if m is None else CodesRustArrayLayer(m, s, commitment),
                InitialStateRustArrayLayer(s),
            )
            for s, m, commitment in sizes
        ]
        self._commitments = [0.0 if f.codes is None else f.codes.commitment for f in self.factors]
        self.consolidation: Consolidation = {}
        self.temperatures: list[float | None] = [None] * len(sizes)
        self.soft = False  # whether forward_batch's codes are soft (the per-layer training step only)
        self.continuous: list[bool] = [False] * len(sizes)  # each snapped factor run unsnapped (the warm-up)
        self.reading = False  # whether an inference pass reads the continuous factors unsnapped (the screen)
        self.couplings: list[float] = [1.0] * len(sizes)  # each factor's next state's share of its code (D25)
        self._free: list[bool] = [False] * len(sizes)  # the last step's unsnapped factors

    @property
    def commitments(self) -> list[float]:
        """Each factor's codes' commitment, 0 for a continuous factor."""
        return self._commitments

    def set_commitments(self, commitments: Sequence[float]) -> None:
        """Each coded factor's commitment from the next training step on (a continuous factor's 0)."""
        set_commitments(self.factors, commitments)
        self._commitments = [0.0 if f.codes is None else f.codes.commitment for f in self.factors]

    def set_temperatures(self, temperatures: Sequence[float | None]) -> None:
        """Each factor's codes' temperature in training from the next step on (None: hard)."""
        check_temperatures(self.factors, temperatures)
        self.temperatures = list(temperatures)

    def set_continuous(self, continuous: Sequence[bool]) -> None:
        """
        Each snapped factor run unsnapped from the next training step on, its cell's output passed on
        and its codes unread (the state formation workplan's warm-up, D4); in inference too when
        `reading` (the screen). The initial state stays snapped (D11).
        """
        check_continuous(self.factors, continuous, self.temperatures)
        self.continuous = list(continuous)

    def set_couplings(self, couplings: Sequence[float]) -> None:
        """
        Each factor's next state `(1 - l) h + l q` from the next training step on (the state formation
        workplan's gradual handover, D25: 1, its code; 0, its cell's output, its codes still learning
        from it); in inference too when `reading` (the screen). The initial state stays snapped.
        """
        check_couplings(self.factors, couplings, self.continuous)
        self.couplings = list(couplings)

    def parts(self) -> list[Any]:
        """The parts with parameters, factor by factor: pooling, cell, codes (if any), initial state."""
        return [
            part for f in self.factors for part in (f.pooling, f.cell, f.codes, f.initial_state) if part is not None
        ]

    def draw(self, rng: pa.Generator) -> None:
        """Each factor's cell's gates, then its codes, factor by factor."""
        for f in self.factors:
            f.cell.draw(rng)
            if f.codes is not None:
                f.codes.draw(rng)

    def _codes(self) -> list[pa.Array | None]:
        return [None if f.codes is None else f.codes.E for f in self.factors]

    def _grad_codes(self) -> list[pa.Array | None]:
        return [None if f.codes is None else f.codes.grad_E for f in self.factors]

    def _targets(self) -> list[pa.Array | None] | None:
        # each factor's commitment target rows after the last snap, from the consolidation (None:
        # every factor's own code)
        if not self.consolidation:
            return None
        codes = self._codes()
        assert all(e is not None for e in codes), "a consolidation needs every factor snapped"
        targets = commitment_targets(self.consolidation, [i for i in self.indices if i is not None])
        assert targets is not None
        return [e.take_rows(t) for e, t in zip(codes, targets, strict=True) if e is not None]

    def _set_grad_codes(self, grads: list[pa.Array | None], free: Sequence[bool] | None = None) -> None:
        # a factor run continuous (free) leaves its codes' gradient as it was
        for f, grad, unsnapped in zip(self.factors, grads, free or [False] * len(grads), strict=True):
            if f.codes is not None and not unsnapped:
                assert grad is not None
                f.codes.grad_E = grad

    def first_state(self, examples: int) -> pa.Array:
        """The initial state, snapped, for each of examples examples, (examples, state_size)."""
        starts = [pa.Array.zeros((examples, f.initial_state.state_size)) + f.initial_state.c for f in self.factors]
        self.H = starts[0] if len(starts) == 1 else pa.concatenate_columns(starts)
        self._Q, self.indices = pa.factored_codes_forward(self.H, self.sizes, self._codes())
        return self._Q

    def first_state_backward(self, dC: pa.Array, weight: float) -> None:
        """After first_state: the initial states' gradients accumulated, through the codes."""
        dH, grads = pa.factored_codes_backward(
            dC, self.H, self._Q, self.sizes, self.indices, self._grad_codes(), weight, self._commitments, self._targets()
        )
        self._set_grad_codes(grads)
        for f, start, size in zip(self.factors, self.starts, self.sizes, strict=True):
            initial = f.initial_state
            initial.grad_c = initial.grad_c + pa.sum_axis0(dH[:, start : start + size])

    def forward_batch(self, X: pa.Array | Sequence[pa.Array], C: pa.Array) -> pa.Array:
        """
        The tokens the output layer reads, X (N, T * d), or each factor's, one a factor (the window
        branches workplan), and the state C (N, S), to the next state (N, S).
        """
        factors = self.factors
        self._free = [c and (self.soft or self.reading) for c in self.continuous]
        self._step = pa.recurrence_step_forward(
            X if isinstance(X, pa.Array) else list(X),
            C,
            [f.pooling.q for f in factors],
            [f.cell.W for f in factors],
            [f.cell.U for f in factors],
            [f.cell.b for f in factors],
            [None if free else e for e, free in zip(self._codes(), self._free, strict=True)],
            self.tokens,
            cross=self.cross,
            temperatures=self.temperatures if self.soft else None,
            couplings=self.couplings if (self.soft or self.reading) and min(self.couplings) < 1.0 else None,
        )
        self.H, self.indices = self._step.h, self._step.indices
        return self._step.state

    def backward_batch(self, dQ: pa.Array, weight: float) -> tuple[pa.Array | list[pa.Array], pa.Array]:
        """
        The next state's gradient (N, S) to the tokens' (N, T * d), each factor's poolings' summed in
        factor order, or each factor's apart if forward_batch read one a factor, and the state's (N,
        S); the parts' accumulated.
        """
        factors = self.factors
        dX, dC, grad_q, grad_W, grad_U, grad_b, grad_E = pa.recurrence_step_backward(
            self._step,
            dQ,
            [f.pooling.q for f in factors],
            [f.cell.W for f in factors],
            [f.cell.U for f in factors],
            [f.pooling.grad_q for f in factors],
            [f.cell.grad_W for f in factors],
            [f.cell.grad_U for f in factors],
            [f.cell.grad_b for f in factors],
            [None if free else e for e, free in zip(self._grad_codes(), self._free, strict=True)],
            weight,
            self._commitments,
            self._targets(),
        )
        for f, gq, gW, gU, gb in zip(factors, grad_q, grad_W, grad_U, grad_b, strict=True):
            f.pooling.grad_q = gq
            f.cell.grad_W, f.cell.grad_U, f.cell.grad_b = gW, gU, gb
        self._set_grad_codes(grad_E, self._free)
        return dX, dC
