"""
A window-recurrent network's layers in pure Python (the window-recurrent adder workplan, stage 6),
the counterparts of recurrent_array_layer.py: the StateInputLayer, which stands in the window's
layer list, and the parts the network owns beside it, the attention pooling, the GRU cell, the
codes and the initial state. Every sum is a left fold from 0.0 in index order (fold), the
products' included, so pure Python agrees with numpy within the products' rounding, and by bits
where numpy folds too (the codes' distances, the pooling's softmax sum).

The StateInputLayer is a layer like the window's others, driven by the layer-major batch path
(layer_major.py), which tells it each example's index (begin_example) so it adds that example's
state; it keeps each example's state gradient for the network to read after the backward pass.
The network's own parts work on a batch, a list of rows, as numpy's do on arrays: forward_batch,
then backward_batch with the gradient of what forward_batch gave, which accumulates the
parameters' gradients and returns the inputs'. Each keeps one forward pass's values. Their
weights are weight sets (token_layer.py's rows), which the optimizer steps as any layer's.

The pooling's softmax and the cell's sigmoid and tanh go through this module's exp, which tests may
replace, as attention_layer's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from indrajala_ml.model.layers.python.batch_norm_layer import fold
from indrajala_ml.model.layers.python.fan_in_aware_init import fan_in_aware_weights, fan_in_aware_weights_and_biases
from indrajala_ml.model.layers.python.residual_layer import ParameterFreeLayer
from indrajala_ml.model.layers.python.token_layer import EmbeddingRow, PositionRow, TokenNode, WeightRow
from indrajala_ml.model.protocols.layer_protocols import InputLayer, TrainableLayer
from indrajala_ml.pcg64 import Pcg64Generator

exp = math.exp

Rows = list[list[float]]


def sigmoid(z: float) -> float:
    """1 / (1 + exp(-z)) through this module's exp, 0.0 where exp(-z) overflows, as numpy's inf gives."""
    try:
        return 1.0 / (1.0 + exp(-z))
    except OverflowError:
        return 0.0


def tanh(z: float) -> float:
    """1 - 2 / (exp(2z) + 1) (W15), numpy's form: exactly 1.0 where exp(2z) overflows."""
    try:
        return 1.0 - 2.0 / (exp(2.0 * z) + 1.0)
    except OverflowError:
        return 1.0


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return fold([x * y for x, y in zip(a, b, strict=True)])


def _column_sums(rows: Sequence[Sequence[float]]) -> list[float]:
    # each column's sum over the rows, a left fold in row order: numpy's sum_rows
    return [fold(column) for column in zip(*rows, strict=True)]


def _add_into(accum: list[float], values: Sequence[float]) -> None:
    for j, value in enumerate(values):
        accum[j] += value


class ProjectionRow(PositionRow):
    """One row of a StateInput's P: weights without a bias, decayed as a linear layer's node."""

    weights_decayed: ClassVar[bool] = True


class StateInputLayer(ParameterFreeLayer[TokenNode]):
    """
    A window's state entering its tokens (W1b-ii, W18): out_t = x_t + P c for each of the tokens
    tokens, P (features, state_size), one weight set per row, c the example's state, from the
    batch's states (states, set by the network before the window's forward pass) at the index
    begin_example gave. Its delta is its downstream; the state's gradient is the delta summed over
    the tokens, then times P (state_deltas, per example), and P's gradient that sum's outer product
    with c. P is drawn as a linear layer's nodes, of fan-in state_size, and decayed.
    """

    # the example the layer reads, its state and its delta's sum over the tokens, per example
    example_fields: ClassVar[tuple[str, ...]] = ("_example", "_state", "_token_sum")

    def __init__(self, input_layer: InputLayer, tokens: int, features: int, state_size: int) -> None:
        assert len(input_layer.nodes) == tokens * features
        self.input_layer = input_layer
        self.tokens = tokens
        self.features = features
        self.state_size = state_size
        self.rows = [ProjectionRow(state_size) for _ in range(features)]
        self.nodes = [TokenNode() for _ in range(tokens * features)]
        self.size = len(self.nodes)
        # the batch's states, which the network sets; each example's state gradient, which it reads
        self.states: Rows = [[0.0] * state_size]
        self.state_deltas: Rows = []
        self._example = 0
        self._state: list[float] = self.states[0]
        self._token_sum: list[float] = []

    def begin_example(self, example: int) -> None:
        """The layer-major path's: the next forward pass is example's, with its state."""
        assert len(self.states[example]) == self.state_size, (
            f"a StateInput needs each example's state, {self.state_size} features; got {self.states[example]}"
        )
        self._example = example
        self._state = self.states[example]
        if example == 0:
            self.state_deltas = [[0.0] * self.state_size for _ in self.states]

    def forward(self) -> None:
        d = self.features
        projected = [_dot(row.weights, self._state) for row in self.rows]
        for i, (node, input_node) in enumerate(zip(self.nodes, self.input_layer.nodes)):
            node.activate(input_node.value() + projected[i % d])

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
        for own_index, node in enumerate(self.nodes):
            node.delta = next_layer.downstream_sum(own_index)
        d = self.features
        self._token_sum = [fold([self.nodes[t * d + j].delta for t in range(self.tokens)]) for j in range(d)]
        self.state_deltas[self._example] = [
            fold([self._token_sum[j] * self.rows[j].weights[k] for j in range(d)]) for k in range(self.state_size)
        ]

    def accumulate_gradients(self) -> None:
        for row, token_sum in zip(self.rows, self._token_sum, strict=True):
            accum = row.weight_gradient_accum
            for k, c_k in enumerate(self._state):
                accum[k] += token_sum * c_k

    def weight_sets(self) -> Sequence[ProjectionRow]:
        return self.rows

    def randomize_fan_in_aware(self, rng: Pcg64Generator) -> None:
        for row, weights in zip(self.rows, fan_in_aware_weights(rng, self.features, self.state_size), strict=True):
            row.set_weights(weights)

    def snapshot_state(self) -> list[tuple[list[float]]]:
        return [(list(row.weights),) for row in self.rows]

    def restore_state(self, layer_snapshot: Sequence[Sequence[list[float]]]) -> None:
        for row, (weights,) in zip(self.rows, layer_snapshot, strict=True):
            row.set_weights(list(weights))


class _Part:
    """What the network's own parts share: their weight sets, saved and stepped as a layer's."""

    def weight_sets(self) -> Sequence[PositionRow]:
        raise NotImplementedError

    def snapshot_state(self) -> list[tuple[list[float], float] | tuple[list[float]]]:
        return [(list(row.weights), row.bias) if row.has_bias else (list(row.weights),) for row in self.weight_sets()]

    def restore_state(self, layer_snapshot: Sequence[Sequence[list[float] | float]]) -> None:
        rows = self.weight_sets()
        assert len(layer_snapshot) == len(rows), f"a {type(self).__name__} restores {len(rows)} rows"
        for row, entry in zip(rows, layer_snapshot, strict=True):
            weights = entry[0]
            assert isinstance(weights, list)
            row.set_weights([float(w) for w in weights])
            if row.has_bias:
                bias = entry[1]
                assert isinstance(bias, float | int)
                row.bias = float(bias)


class AttentionPoolingLayer(_Part):
    """
    The window's summary (W1b-i, W19): one learned query q of features features, one weight set,
    over the tokens tokens the window's output layer reads, s_t = q . x_t, p = softmax(s),
    u = sum_t p_t x_t. q starts at zero, a mean over the tokens, and isn't decayed.
    """

    def __init__(self, tokens: int, features: int) -> None:
        self.tokens = tokens
        self.features = features
        self.query = PositionRow(features)

    def weight_sets(self) -> Sequence[PositionRow]:
        return [self.query]

    def _token_rows(self, x: Sequence[float]) -> Rows:
        d = self.features
        return [list(x[t * d : (t + 1) * d]) for t in range(self.tokens)]

    def forward_batch(self, X: Rows) -> Rows:
        """X, each example's tokens (T * d), to u, each example's summary (d)."""
        q = self.query.weights
        self._X = [self._token_rows(x) for x in X]
        self._p: Rows = []
        U: Rows = []
        for tokens in self._X:
            s = [_dot(x_t, q) for x_t in tokens]
            m = max(s)
            e = [exp(s_t - m) for s_t in s]
            total = fold(e)
            p = [e_t / total for e_t in e]
            self._p.append(p)
            U.append([fold([p_t * x_t[j] for p_t, x_t in zip(p, tokens)]) for j in range(self.features)])
        return U

    def backward_batch(self, dU: Rows) -> Rows:
        """u's gradient to the tokens' (T * d); q's gradient accumulated."""
        q = self.query.weights
        grad_rows: Rows = []
        dX: Rows = []
        for tokens, p, du in zip(self._X, self._p, dU, strict=True):
            dp = [_dot(x_t, du) for x_t in tokens]
            weighted = fold([dp_t * p_t for dp_t, p_t in zip(dp, p)])
            dS = [p_t * (dp_t - weighted) for p_t, dp_t in zip(p, dp)]
            grad_rows.extend([dS_t * x for x in x_t] for dS_t, x_t in zip(dS, tokens))
            dX.append([p_t * du[j] + dS_t * q[j] for p_t, dS_t in zip(p, dS) for j in range(self.features)])
        _add_into(self.query.weight_gradient_accum, _column_sums(grad_rows))
        return dX


class GRULayer(_Part):
    """
    The cell (W1b-iii; Cho et al. 2014), reading the summary u (input_size) and the state c
    (state_size), its gates in the order r, z, h, each state_size rows, each row a weight set of
    W's row then U's (input_size + state_size weights) and b's value:
    r = sigmoid(W_r u + U_r c + b_r), z = sigmoid(W_z u + U_z c + b_z),
    h~ = tanh(W_h u + U_h (r * c) + b_h), h = z * c + (1 - z) * h~.
    Each gate's rows are drawn as a dense layer's nodes of fan-in d + s, then b_z set to
    update_bias, toward keeping the state (W13). The weights are decayed, the biases aren't.
    """

    def __init__(self, input_size: int, state_size: int, update_bias: float = 1.0) -> None:
        self.input_size = input_size
        self.state_size = state_size
        self.update_bias = update_bias
        self.rows = [WeightRow(input_size + state_size) for _ in range(3 * state_size)]

    def weight_sets(self) -> Sequence[WeightRow]:
        return self.rows

    def draw(self, rng: Pcg64Generator) -> None:
        s, d = self.state_size, self.input_size
        for gate in range(3):
            weights, biases = fan_in_aware_weights_and_biases(rng, s, d + s)
            for row, row_weights, bias in zip(self.rows[gate * s : (gate + 1) * s], weights, biases, strict=True):
                row.set_weights(row_weights)
                row.bias = bias
        for row in self.rows[s : 2 * s]:
            row.bias = self.update_bias

    def forward_batch(self, U_in: Rows, C: Rows) -> Rows:
        """u (d per example) and c (s per example) to h (s per example)."""
        s, d = self.state_size, self.input_size
        self._u, self._c = U_in, C
        self._r: Rows = []
        self._z: Rows = []
        self._rc: Rows = []
        self._h_tilde: Rows = []
        H: Rows = []
        for u, c in zip(U_in, C, strict=True):
            a = [_dot(u, row.weights[:d]) + row.bias for row in self.rows]
            rz = [a[i] + _dot(c, self.rows[i].weights[d:]) for i in range(2 * s)]
            r, z = [sigmoid(v) for v in rz[:s]], [sigmoid(v) for v in rz[s:]]
            rc = [r_k * c_k for r_k, c_k in zip(r, c)]
            h_tilde = [tanh(a[2 * s + k] + _dot(rc, self.rows[2 * s + k].weights[d:])) for k in range(s)]
            self._r.append(r)
            self._z.append(z)
            self._rc.append(rc)
            self._h_tilde.append(h_tilde)
            H.append([z_k * c_k + (1.0 - z_k) * h_k for z_k, c_k, h_k in zip(z, c, h_tilde)])
        return H

    def backward_batch(self, dH: Rows) -> tuple[Rows, Rows]:
        """h's gradient to u's and c's; the rows' gradients accumulated."""
        s, d = self.state_size, self.input_size
        dU_in: Rows = []
        dC: Rows = []
        dA_rows: Rows = []
        for example, dh in enumerate(dH):
            r, z, h_tilde = self._r[example], self._z[example], self._h_tilde[example]
            c = self._c[example]
            dA_h = [dh[k] * (1.0 - z[k]) * (1.0 - h_tilde[k] * h_tilde[k]) for k in range(s)]
            d_rc = [fold([dA_h[i] * self.rows[2 * s + i].weights[d + k] for i in range(s)]) for k in range(s)]
            dA_r = [d_rc[k] * c[k] * r[k] * (1.0 - r[k]) for k in range(s)]
            dA_z = [dh[k] * (c[k] - h_tilde[k]) * z[k] * (1.0 - z[k]) for k in range(s)]
            dA = [*dA_r, *dA_z, *dA_h]
            dA_rows.append(dA)
            dU_in.append([fold([dA[i] * row.weights[j] for i, row in enumerate(self.rows)]) for j in range(d)])
            dC.append(
                [
                    (dh[k] * z[k] + d_rc[k] * r[k]) + fold([dA[i] * self.rows[i].weights[d + k] for i in range(2 * s)])
                    for k in range(s)
                ]
            )
        # each row's gradient over the batch, then added, as numpy's products then +=
        for i, row in enumerate(self.rows):
            inputs = [[*u, *(c if i < 2 * s else rc)] for u, c, rc in zip(self._u, self._c, self._rc, strict=True)]
            _add_into(
                row.weight_gradient_accum,
                [fold([dA[i] * x[j] for dA, x in zip(dA_rows, inputs)]) for j in range(d + s)],
            )
            row.bias_gradient_accum += fold([dA[i] for dA in dA_rows])
        return dU_in, dC


class CodesLayer(_Part):
    """
    The state's discretiser (W4, W14; van den Oord et al. 2017): count learned codes, each a weight
    set of size weights, each h snapped to its nearest code e by squared distance, the lowest index
    on a tie; the distance a left fold over the features of (h - e)^2, numpy's. The forward pass
    gives e; backward_batch passes the gradient straight through to h, plus the commitment loss's,
    and gives the codes the codebook loss's: per example and window, weight * (||sg[h] - e||^2 +
    commitment * ||h - sg[e]||^2). The codes are drawn as a linear layer's nodes of fan-in size,
    and not decayed.
    """

    def __init__(self, count: int, size: int, commitment: float = 0.25) -> None:
        self.count = count
        self.size = size
        self.commitment = commitment
        self.codes = [EmbeddingRow(size) for _ in range(count)]

    def weight_sets(self) -> Sequence[EmbeddingRow]:
        return self.codes

    def draw(self, rng: Pcg64Generator) -> None:
        for code, weights in zip(self.codes, fan_in_aware_weights(rng, self.count, self.size), strict=True):
            code.set_weights(weights)

    def nearest(self, h: Sequence[float]) -> int:
        """h's nearest code's index."""
        distances = [fold([(h_j - e_j) * (h_j - e_j) for h_j, e_j in zip(h, code.weights)]) for code in self.codes]
        return distances.index(min(distances))

    def forward_batch(self, H: Rows) -> Rows:
        """h (size per example) to its nearest code, new lists."""
        self._H = H
        self.indices = [self.nearest(h) for h in H]
        self._Q = [list(self.codes[index].weights) for index in self.indices]
        return [list(q) for q in self._Q]

    def loss_batch(self, weight: float) -> float:
        """The batch's code losses, weighted, summed over the examples: what backward_batch differentiates."""
        squared = fold([fold([(h_j - q_j) * (h_j - q_j) for h_j, q_j in zip(h, q)]) for h, q in zip(self._H, self._Q)])
        return weight * (1.0 + self.commitment) * squared

    def backward_batch(self, dQ: Rows, weight: float) -> Rows:
        """The code's gradient to h's: straight through, plus the commitment loss's; the codes' accumulated."""
        dH: Rows = []
        for h, q, index, dq in zip(self._H, self._Q, self.indices, dQ, strict=True):
            differences = [h_j - q_j for h_j, q_j in zip(h, q)]
            accum = self.codes[index].weight_gradient_accum
            for j, difference in enumerate(differences):
                accum[j] += -2.0 * weight * difference
            dH.append(
                [dq_j + (2.0 * weight * self.commitment) * difference for dq_j, difference in zip(dq, differences)]
            )
        return dH


class InitialStateLayer(_Part):
    """
    The first window's state before it is snapped (W8): a learned vector c of state_size features,
    one weight set, the same for every example, starting at zero and not decayed. Its gradient is
    the snapped state's summed over the examples.
    """

    def __init__(self, state_size: int) -> None:
        self.state_size = state_size
        self.state = PositionRow(state_size)

    def weight_sets(self) -> Sequence[PositionRow]:
        return [self.state]

    def forward_batch(self, examples: int) -> Rows:
        """c for each of examples examples, new lists."""
        return [list(self.state.weights) for _ in range(examples)]

    def backward_batch(self, dC: Rows) -> None:
        _add_into(self.state.weight_gradient_accum, _column_sums(dC))


class _FactorParts:
    """One factor's parts: its pooling, cell, codes (None: continuous) and initial state."""

    def __init__(
        self, pooling: AttentionPoolingLayer, cell: GRULayer, codes: CodesLayer | None, initial_state: InitialStateLayer
    ) -> None:
        self.pooling = pooling
        self.cell = cell
        self.codes = codes
        self.initial_state = initial_state


class RecurrenceLayer:
    """
    RecurrenceArrayLayer in pure Python (the factored-state workplan, stage 4): a window's
    recurrence over its state's factors, each factor's pooling, cell and codes, factor by factor,
    on rows; factor g's state the row's s_g values after the factors before it, its cell's input
    [u_g, c_(-g)] with cross. The factors' shared sums are in factor order from factor 1's term,
    as numpy's.
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
                AttentionPoolingLayer(tokens, features),
                GRULayer(features + (self.state_size - s if cross else 0), s, update_bias),
                None if m is None else CodesLayer(m, s, commitment),
                InitialStateLayer(s),
            )
            for s, m, commitment in sizes
        ]

    def parts(self) -> list[_Part]:
        """The parts with weights, factor by factor: pooling, cell, codes (if any), initial state."""
        return [
            part for f in self.factors for part in (f.pooling, f.cell, f.codes, f.initial_state) if part is not None
        ]

    def draw(self, rng: Pcg64Generator) -> None:
        """Each factor's cell's gates, then its codes, factor by factor."""
        for f in self.factors:
            f.cell.draw(rng)
            if f.codes is not None:
                f.codes.draw(rng)

    def _block(self, C: Rows, g: int) -> Rows:
        start, size = self.starts[g], self.sizes[g]
        return [list(c[start : start + size]) for c in C]

    def _others(self, c: Sequence[float], g: int) -> list[float]:
        start, size = self.starts[g], self.sizes[g]
        return [*c[:start], *c[start + size :]]

    def _snapped(self, hs: list[Rows]) -> Rows:
        self._hs = hs
        qs = [h if f.codes is None else f.codes.forward_batch(h) for f, h in zip(self.factors, hs, strict=True)]
        return [[v for q in rows for v in q] for rows in zip(*qs, strict=True)]

    @property
    def H(self) -> Rows:
        """The last forward pass's cells' outputs (or initial states) before they were snapped."""
        return [[v for h in rows for v in h] for rows in zip(*self._hs, strict=True)]

    @property
    def indices(self) -> list[list[int] | None]:
        """The last forward pass's code indices, per factor; None for a continuous factor."""
        return [None if f.codes is None else f.codes.indices for f in self.factors]

    def _snap_backward(self, dQ: Rows, weight: float) -> list[Rows]:
        return [
            self._block(dQ, g) if f.codes is None else f.codes.backward_batch(self._block(dQ, g), weight)
            for g, f in enumerate(self.factors)
        ]

    def first_state(self, examples: int) -> Rows:
        """The initial state, snapped, for each of examples examples."""
        return self._snapped([f.initial_state.forward_batch(examples) for f in self.factors])

    def first_state_backward(self, dC: Rows, weight: float) -> None:
        """After first_state: the initial states' gradients accumulated, through the codes."""
        for f, dH in zip(self.factors, self._snap_backward(dC, weight), strict=True):
            f.initial_state.backward_batch(dH)

    def forward_batch(self, X: Rows, C: Rows) -> Rows:
        """The tokens the output layer reads, X (T * d per example), and the state C, to the next state."""
        hs: list[Rows] = []
        for g, f in enumerate(self.factors):
            U = f.pooling.forward_batch(X)
            inputs = [[*u, *self._others(c, g)] for u, c in zip(U, C, strict=True)] if self.cross else U
            hs.append(f.cell.forward_batch(inputs, self._block(C, g)))
        return self._snapped(hs)

    def backward_batch(self, dQ: Rows, weight: float) -> tuple[Rows, Rows]:
        """The next state's gradient to the tokens' (T * d per example) and the state's; the parts' accumulated."""
        d = self.features
        dX: Rows | None = None
        dC_own: list[Rows] = []
        dC_cross: list[Rows] = []
        for f, dH in zip(self.factors, self._snap_backward(dQ, weight), strict=True):
            d_inputs, dC = f.cell.backward_batch(dH)
            if self.cross:
                dC_cross.append([row[d:] for row in d_inputs])
                d_inputs = [row[:d] for row in d_inputs]
            dX_g = f.pooling.backward_batch(d_inputs)
            dX = (
                dX_g
                if dX is None
                else [[a + b for a, b in zip(x, y, strict=True)] for x, y in zip(dX, dX_g, strict=True)]
            )
            dC_own.append(dC)
        dC = [[v for block in rows for v in block] for rows in zip(*dC_own, strict=True)]
        for g, piece in enumerate(dC_cross):
            start, size = self.starts[g], self.sizes[g]
            for row, values in zip(dC, piece, strict=True):
                targets = [*range(start), *range(start + size, self.state_size)]
                for j, value in zip(targets, values, strict=True):
                    row[j] += value
        assert dX is not None
        return dX, dC
