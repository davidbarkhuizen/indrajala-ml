# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, U, X, H, which strict mode takes for constants)
"""
A window-recurrent network's layers in numpy (the window-recurrent adder workplan, stage 3):
the StateInput, which stands in the window's layer list, and the parts the network owns beside
it, the attention pooling, the GRU cell and the codes, which RecurrenceArrayLayer runs for each of
the state's factors (the factored-state workplan, stage 4). Every sum but a product is a left fold
through np.cumsum (tests/model/layers/test_summation_order.py), as attention's.

The StateInput is a layer like the window's others, driven through the list (BatchShaped); the
network sets its state before the window's forward pass and reads the state's gradient after the
backward pass. The network's own parts take and give their gradients explicitly: forward_batch,
then backward_batch with the gradient of what forward_batch gave, which accumulates the
parameters' gradients and returns the inputs'. Each keeps one forward pass's values, so a network
running several windows re-runs a window's forward pass before its backward pass.

The softmax's exp and the cell's tanh go through this module's exp, which tests may replace with
another implementation's, as attention's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from indrajala_ml.model.layers.numpy.array_layer import (
    FloatArray,
    fan_in_aware_random_layer,
    fan_in_aware_random_weights,
)
from indrajala_ml.model.layers.numpy.batch_norm_array_layer import sum_rows
from indrajala_ml.model.layers.numpy.token_array_layer import BatchShaped

exp = np.exp


def sigmoid(z: FloatArray) -> FloatArray:
    """1 / (1 + exp(-z)), through this module's exp; array_layer.sigmoid's form."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + exp(-z))


def tanh(z: FloatArray) -> FloatArray:
    """
    1 - 2 / (exp(2z) + 1) (the window-recurrent adder workplan, W15): through exp, as sigmoid, so
    the three implementations agree where their exp does. A large z overflows exp(2z) to inf and
    gives exactly 1; a large negative z gives exactly -1.
    """
    with np.errstate(over="ignore"):
        return 1.0 - 2.0 / (exp(2.0 * z) + 1.0)


def _fold(values: FloatArray, axis: int) -> FloatArray:
    # the sum along axis, a left fold
    return np.take(np.cumsum(values, axis=axis), -1, axis=axis)


class StateInputArrayLayer(BatchShaped):
    """
    A window's state entering its tokens (the window-recurrent adder workplan, W1b-ii, W18):
    out_t = x_t + P c for each of the tokens tokens, P (features, state_size), c the example's
    state, set by set_state before the forward pass. Its delta is its downstream; the state's
    gradient is the delta summed over the tokens, then times P, and P's gradient that sum's outer
    product with c, summed over the examples. P is drawn as a dense layer's W of fan-in state_size,
    and decayed.
    """

    decayed: ClassVar[tuple[bool, ...]] = (True,)

    def __init__(self, tokens: int, features: int, state_size: int) -> None:
        self.tokens = tokens
        self.features = features
        self.state_size = state_size
        self.size = tokens * features
        self.input_size = self.size
        self.P: FloatArray = np.zeros((features, state_size))
        self.grad_P: FloatArray = np.zeros((features, state_size))
        self.state: FloatArray = np.zeros((1, state_size))

    def parameters(self) -> tuple[FloatArray, ...]:
        return (self.P,)

    def gradients(self) -> tuple[FloatArray, ...]:
        return (self.grad_P,)

    def set_parameters(self, parameters: Sequence[FloatArray]) -> None:
        (self.P,) = parameters

    def reset_gradient_accum(self) -> None:
        self.grad_P = np.zeros(self.P.shape)

    def draw(self, rng: np.random.Generator) -> None:
        self.P = fan_in_aware_random_weights(rng, self.features, self.state_size)

    def set_state(self, state: FloatArray) -> None:
        """The batch's states, (N, state_size), which the next forward pass adds."""
        self.state = state

    def forward_batch(self, X: FloatArray) -> FloatArray:
        n = X.shape[0]
        assert self.state.shape == (n, self.state_size), (
            f"a StateInput needs the batch's states, ({n}, {self.state_size}); got {self.state.shape}"
        )
        projected = self.state @ self.P.T
        return (X.reshape(n, self.tokens, self.features) + projected[:, np.newaxis, :]).reshape(X.shape)

    def _backward(self, downstream: FloatArray) -> None:
        self.delta_batch = downstream
        n = downstream.shape[0]
        # (N, d): each example's delta summed over its tokens
        self._token_sum = _fold(downstream.reshape(n, self.tokens, self.features), axis=1)
        self.state_delta = self._token_sum @ self.P

    def downstream_batch(self) -> FloatArray:
        return self.delta_batch

    def accumulate_gradient_batch(self, input_activation_batch: FloatArray) -> None:
        self.grad_P += self._token_sum.T @ self.state


class AttentionPoolingArrayLayer:
    """
    The window's summary (the window-recurrent adder workplan, W1b-i, W19): one learned query q of
    features features over the tokens tokens the window's output layer reads, s_t = q . x_t,
    p = softmax(s), u = sum_t p_t x_t. q starts at zero, a mean over the tokens, and isn't decayed.
    """

    decayed: ClassVar[tuple[bool, ...]] = (False,)

    def __init__(self, tokens: int, features: int) -> None:
        self.tokens = tokens
        self.features = features
        self.q: FloatArray = np.zeros(features)
        self.grad_q: FloatArray = np.zeros(features)

    def parameters(self) -> tuple[FloatArray, ...]:
        return (self.q,)

    def gradients(self) -> tuple[FloatArray, ...]:
        return (self.grad_q,)

    def set_parameters(self, parameters: Sequence[FloatArray]) -> None:
        (self.q,) = parameters

    def reset_gradient_accum(self) -> None:
        self.grad_q = np.zeros(self.q.shape)

    def forward_batch(self, X: FloatArray) -> FloatArray:
        """X (N, T * d), the tokens, to u (N, d)."""
        self._X = X.reshape(X.shape[0], self.tokens, self.features)
        S = self._X @ self.q
        e = exp(S - S.max(axis=1, keepdims=True))
        self._p = e / _fold(e, axis=1)[:, np.newaxis]
        return (self._p[:, np.newaxis, :] @ self._X)[:, 0, :]

    def backward_batch(self, dU: FloatArray) -> FloatArray:
        """u's gradient (N, d) to the tokens' (N, T * d); q's gradient accumulated."""
        X, p = self._X, self._p
        dp = (X @ dU[:, :, np.newaxis])[:, :, 0]
        dS = p * (dp - _fold(dp * p, axis=1)[:, np.newaxis])
        self.grad_q += sum_rows(dS.reshape(-1)[:, np.newaxis] * X.reshape(-1, self.features))
        dX = p[:, :, np.newaxis] * dU[:, np.newaxis, :] + dS[:, :, np.newaxis] * self.q
        return dX.reshape(dX.shape[0], -1)


class GRUArrayLayer:
    """
    The cell (the window-recurrent adder workplan, W1b-iii; Cho et al. 2014), reading the summary u
    (input_size) and the state c (state_size), its gates in the order r, z, h, each a block of
    state_size rows of W (3s, d), U (3s, s) and b (3s):
    r = sigmoid(W_r u + U_r c + b_r), z = sigmoid(W_z u + U_z c + b_z),
    h~ = tanh(W_h u + U_h (r * c) + b_h), h = z * c + (1 - z) * h~.
    Each gate's [W U] and b are drawn as a dense layer's of fan-in d + s, then b_z set to
    update_bias, toward keeping the state (W13). W and U are decayed, b isn't.
    """

    decayed: ClassVar[tuple[bool, ...]] = (True, True, False)

    def __init__(self, input_size: int, state_size: int, update_bias: float = 1.0) -> None:
        self.input_size = input_size
        self.state_size = state_size
        self.update_bias = update_bias
        rows = 3 * state_size
        self.W: FloatArray = np.zeros((rows, input_size))
        self.U: FloatArray = np.zeros((rows, state_size))
        self.b: FloatArray = np.zeros(rows)
        self.reset_gradient_accum()

    def parameters(self) -> tuple[FloatArray, ...]:
        return self.W, self.U, self.b

    def gradients(self) -> tuple[FloatArray, ...]:
        return self.grad_W, self.grad_U, self.grad_b

    def set_parameters(self, parameters: Sequence[FloatArray]) -> None:
        self.W, self.U, self.b = parameters

    def reset_gradient_accum(self) -> None:
        self.grad_W = np.zeros(self.W.shape)
        self.grad_U = np.zeros(self.U.shape)
        self.grad_b = np.zeros(self.b.shape)

    def draw(self, rng: np.random.Generator) -> None:
        s, d = self.state_size, self.input_size
        gates = [fan_in_aware_random_layer(rng, s, d + s) for _ in range(3)]
        self.W = np.concatenate([W[:, :d] for W, _ in gates])
        self.U = np.concatenate([W[:, d:] for W, _ in gates])
        self.b = np.concatenate([b for _, b in gates])
        self.b[s : 2 * s] = self.update_bias

    def forward_batch(self, U_in: FloatArray, C: FloatArray) -> FloatArray:
        """u (N, d) and c (N, s) to h (N, s)."""
        s = self.state_size
        self._u, self._c = U_in, C
        # the input's part of every gate, then the state's of r and z
        A = U_in @ self.W.T + self.b
        rz = A[:, : 2 * s] + C @ self.U[: 2 * s].T
        self._r, self._z = sigmoid(rz[:, :s]), sigmoid(rz[:, s:])
        self._rc = self._r * C
        self._h_tilde = tanh(A[:, 2 * s :] + self._rc @ self.U[2 * s :].T)
        return self._z * C + (1.0 - self._z) * self._h_tilde

    def backward_batch(self, dH: FloatArray) -> tuple[FloatArray, FloatArray]:
        """h's gradient (N, s) to u's (N, d) and c's (N, s); the gradients of W, U and b accumulated."""
        s = self.state_size
        r, z, h_tilde, C = self._r, self._z, self._h_tilde, self._c
        dA_h = dH * (1.0 - z) * (1.0 - h_tilde * h_tilde)
        d_rc = dA_h @ self.U[2 * s :]
        dA_r = d_rc * C * r * (1.0 - r)
        dA_z = dH * (C - h_tilde) * z * (1.0 - z)
        dA = np.concatenate([dA_r, dA_z, dA_h], axis=1)
        self.grad_W += dA.T @ self._u
        self.grad_U[: 2 * s] += dA[:, : 2 * s].T @ C
        self.grad_U[2 * s :] += dA_h.T @ self._rc
        self.grad_b += sum_rows(dA)
        dU_in = dA @ self.W
        dC = (dH * z + d_rc * r) + dA[:, : 2 * s] @ self.U[: 2 * s]
        return dU_in, dC


class CodesArrayLayer:
    """
    The state's discretiser (the window-recurrent adder workplan, W4, W14; van den Oord et al.
    2017): count learned codes E (count, size), each h snapped to its nearest code e by squared
    distance, the lowest index on a tie; the distance a left fold over the features of (h - e)^2,
    so every implementation picks the same code. The forward pass gives e; backward_batch passes
    the gradient straight through to h, plus the commitment loss's, and gives E the codebook
    loss's: per example and window, weight * (||sg[h] - e||^2 + commitment * ||h - sg[e]||^2),
    weight the network's (1 / its window count, W14). E is drawn as a dense layer's W of fan-in
    size, and not decayed.
    """

    decayed: ClassVar[tuple[bool, ...]] = (False,)

    def __init__(self, count: int, size: int, commitment: float = 0.25) -> None:
        self.count = count
        self.size = size
        self.commitment = commitment
        self.E: FloatArray = np.zeros((count, size))
        self.grad_E: FloatArray = np.zeros((count, size))

    def parameters(self) -> tuple[FloatArray, ...]:
        return (self.E,)

    def gradients(self) -> tuple[FloatArray, ...]:
        return (self.grad_E,)

    def set_parameters(self, parameters: Sequence[FloatArray]) -> None:
        (self.E,) = parameters

    def reset_gradient_accum(self) -> None:
        self.grad_E = np.zeros(self.E.shape)

    def draw(self, rng: np.random.Generator) -> None:
        self.E = fan_in_aware_random_weights(rng, self.count, self.size)

    def nearest(self, H: FloatArray) -> npt.NDArray[np.intp]:
        """Each row of H's nearest code's index."""
        differences = H[:, np.newaxis, :] - self.E[np.newaxis, :, :]
        distances = _fold(differences * differences, axis=2)
        return np.argmin(distances, axis=1)

    def forward_batch(self, H: FloatArray) -> FloatArray:
        """h (N, size) to its nearest code (N, size), a new array."""
        self._H = H
        self.indices = self.nearest(H)
        self._Q = self.E[self.indices]
        return self._Q

    def loss_batch(self, weight: float) -> float:
        """The batch's code losses, weighted, summed over the examples: what backward_batch differentiates."""
        differences = self._H - self._Q
        squared = _fold(_fold(differences * differences, axis=1), axis=0)
        return float(weight * (1.0 + self.commitment) * squared)

    def backward_batch(self, dQ: FloatArray, weight: float) -> FloatArray:
        """The code's gradient (N, size) to h's: straight through, plus the commitment loss's; E's accumulated."""
        differences = self._H - self._Q
        np.add.at(self.grad_E, self.indices, -2.0 * weight * differences)
        return dQ + (2.0 * weight * self.commitment) * differences


class InitialStateArrayLayer:
    """
    The first window's state before it is snapped (the window-recurrent adder workplan, W8): a
    learned vector c of state_size features, the same for every example, starting at zero and not
    decayed. Its gradient is the snapped state's gradient summed over the examples.
    """

    decayed: ClassVar[tuple[bool, ...]] = (False,)

    def __init__(self, state_size: int) -> None:
        self.state_size = state_size
        self.c: FloatArray = np.zeros(state_size)
        self.grad_c: FloatArray = np.zeros(state_size)

    def parameters(self) -> tuple[FloatArray, ...]:
        return (self.c,)

    def gradients(self) -> tuple[FloatArray, ...]:
        return (self.grad_c,)

    def set_parameters(self, parameters: Sequence[FloatArray]) -> None:
        (self.c,) = parameters

    def reset_gradient_accum(self) -> None:
        self.grad_c = np.zeros(self.c.shape)

    def forward_batch(self, examples: int) -> FloatArray:
        """c for each of examples examples, (examples, state_size), a new array."""
        return np.tile(self.c, (examples, 1))

    def backward_batch(self, dC: FloatArray) -> None:
        self.grad_c += sum_rows(dC)


class _FactorParts:
    """One factor's parts: its pooling, cell, codes (None: continuous) and initial state."""

    def __init__(
        self,
        pooling: AttentionPoolingArrayLayer,
        cell: GRUArrayLayer,
        codes: CodesArrayLayer | None,
        initial_state: InitialStateArrayLayer,
    ) -> None:
        self.pooling = pooling
        self.cell = cell
        self.codes = codes
        self.initial_state = initial_state


class RecurrenceArrayLayer:
    """
    A window's recurrence over its state's factors (the factored-state workplan, stage 4;
    window_recurrence.py): each factor's pooling, cell and codes, factor by factor, the crate's
    recurrence_step_* in numpy (src/factored.rs). Factor g's state is the state's s_g columns after
    the factors before it; with cross, its cell's input is [u_g, c_(-g)], the other factors'
    states in factor order after its summary. The factors' shared sums are in factor order from
    factor 1's term: the tokens' gradient, every factor's pooling's, and with cross, each factor's
    state gradient, its own cell's, then each other factor's through its c_(-h) columns. With one
    factor, this is the window's step as it was before factors, by bits.

    forward_batch gives the next state, keeping each factor's indices (None for a continuous factor)
    and the cells' outputs before snapping, H; backward_batch, after it, gives the tokens' and the
    state's gradients and accumulates the parts'. first_state and first_state_backward do the same
    for the learned initial state, snapped (W8).
    """

    def __init__(
        self,
        sizes: Sequence[tuple[int, int | None, float]],
        tokens: int,
        features: int,
        cross: bool,
        update_bias: float,
    ) -> None:
        """sizes: each factor's (state size, code count or None, commitment)."""
        self.sizes = [s for s, _, _ in sizes]
        self.cross = cross
        self.features = features
        self.state_size = sum(self.sizes)
        self.starts = [sum(self.sizes[:g]) for g in range(len(self.sizes))]
        self.factors = [
            _FactorParts(
                AttentionPoolingArrayLayer(tokens, features),
                GRUArrayLayer(features + (self.state_size - s if cross else 0), s, update_bias),
                None if m is None else CodesArrayLayer(m, s, commitment),
                InitialStateArrayLayer(s),
            )
            for s, m, commitment in sizes
        ]

    def parts(self) -> list[Any]:
        """The parts with parameters, factor by factor: pooling, cell, codes (if any), initial state."""
        return [
            part for f in self.factors for part in (f.pooling, f.cell, f.codes, f.initial_state) if part is not None
        ]

    def draw(self, rng: np.random.Generator) -> None:
        """Each factor's cell's gates, then its codes, factor by factor."""
        for f in self.factors:
            f.cell.draw(rng)
            if f.codes is not None:
                f.codes.draw(rng)

    def _block(self, C: FloatArray, g: int) -> FloatArray:
        return C[:, self.starts[g] : self.starts[g] + self.sizes[g]]

    def _others(self, C: FloatArray, g: int) -> FloatArray:
        start, size = self.starts[g], self.sizes[g]
        return np.concatenate([C[:, :start], C[:, start + size :]], axis=1)

    def _snapped(self, hs: list[FloatArray]) -> FloatArray:
        self._hs = hs
        return np.concatenate(
            [h if f.codes is None else f.codes.forward_batch(h) for f, h in zip(self.factors, hs, strict=True)], axis=1
        )

    @property
    def H(self) -> FloatArray:
        """The last forward pass's cells' outputs (or initial states) before they were snapped, (N, S)."""
        return np.concatenate(self._hs, axis=1)

    @property
    def indices(self) -> list[npt.NDArray[np.intp] | None]:
        """The last forward pass's code indices, per factor; None for a continuous factor."""
        return [None if f.codes is None else f.codes.indices for f in self.factors]

    def _snap_backward(self, dQ: FloatArray, weight: float) -> list[FloatArray]:
        return [
            self._block(dQ, g) if f.codes is None else f.codes.backward_batch(self._block(dQ, g), weight)
            for g, f in enumerate(self.factors)
        ]

    def first_state(self, examples: int) -> FloatArray:
        """The initial state, snapped, for each of examples examples, (examples, state_size)."""
        return self._snapped([f.initial_state.forward_batch(examples) for f in self.factors])

    def first_state_backward(self, dC: FloatArray, weight: float) -> None:
        """After first_state: the initial states' gradients accumulated, through the codes."""
        for f, dH in zip(self.factors, self._snap_backward(dC, weight), strict=True):
            f.initial_state.backward_batch(dH)

    def forward_batch(self, X: FloatArray, C: FloatArray) -> FloatArray:
        """The tokens the output layer reads, X (N, T * d), and the state C (N, S), to the next state (N, S)."""
        hs: list[FloatArray] = []
        for g, f in enumerate(self.factors):
            u = f.pooling.forward_batch(X)
            inputs = np.concatenate([u, self._others(C, g)], axis=1) if self.cross else u
            hs.append(f.cell.forward_batch(inputs, self._block(C, g)))
        return self._snapped(hs)

    def backward_batch(self, dQ: FloatArray, weight: float) -> tuple[FloatArray, FloatArray]:
        """The next state's gradient (N, S) to the tokens' (N, T * d) and the state's (N, S); the parts' accumulated."""
        d = self.features
        dX: FloatArray | None = None
        dC_own: list[FloatArray] = []
        dC_cross: list[FloatArray] = []
        for f, dH in zip(self.factors, self._snap_backward(dQ, weight), strict=True):
            d_inputs, dC = f.cell.backward_batch(dH)
            if self.cross:
                dC_cross.append(d_inputs[:, d:])
                d_inputs = d_inputs[:, :d]
            dX_g = f.pooling.backward_batch(d_inputs)
            dX = dX_g if dX is None else dX + dX_g
            dC_own.append(dC)
        dC = np.concatenate(dC_own, axis=1)
        for g, piece in enumerate(dC_cross):
            start, size = self.starts[g], self.sizes[g]
            dC[:, :start] += piece[:, :start]
            dC[:, start + size :] += piece[:, start:]
        assert dX is not None
        return dX, dC
