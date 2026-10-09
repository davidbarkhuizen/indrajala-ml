# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, Q, K, V, P, H, which strict mode takes for constants)
"""
Multi-head self-attention in pure Python (the multi-head attention workplan, D4, D9; README, Layer
norm and attention), the counterpart of attention_array_layer.py, in the same three blocks per
pass: project, attend (each head), combine. Every expression is the README's, per scalar, in its
grouping, and every sum, the products' included, a left fold from 0.0 in index order (fold). One
code path for every head count.

The layer's per-example caches (Q, K, V, P, H, and backward dQ, dK, dV, dX) are its own, not its
nodes': they are matrices over the tokens, packed as the crate's, Q, K, V, H, dQ, dK and dV
(T, h * d_k), head i's columns i * d_k to (i + 1) * d_k - 1, and P (T, h * T), head i's weights
columns i * T to (i + 1) * T - 1. It names them in example_fields, and the layer-major batch path
(layer_major.py) keeps them in lanes, as it does the nodes' fields. Every pass assigns new lists,
never writes into a cached one, so a lane can hold them without a copy.

The softmax's exp is this module's exp, which tests may replace, as attention_array_layer's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from indrajala_ml.model.layers.python.batch_norm_layer import fold
from indrajala_ml.model.layers.python.residual_layer import ParameterFreeLayer
from indrajala_ml.model.layers.python.token_layer import TokenNode, WeightRow, restore_rows, snapshot_rows, token_values
from indrajala_ml.model.protocols.layer_protocols import InputLayer, TrainableLayer
from indrajala_ml.pcg64 import Pcg64Generator

exp = math.exp

Matrix = list[list[float]]


def _affine(X: Matrix, rows: Sequence[WeightRow]) -> Matrix:
    # X W^T + b: each token's product with each row, then the bias
    return [[fold([x_j * w_j for x_j, w_j in zip(x, row.weights)]) + row.bias for row in rows] for x in X]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return fold([a_i * b_i for a_i, b_i in zip(a, b)])


def _transpose(A: Matrix) -> Matrix:
    return [list(column) for column in zip(*A)]


def _times_rows(A: Matrix, rows: Sequence[WeightRow]) -> Matrix:
    # A W, W's rows the weight rows: (A W)_tj = sum_k(A_tk * W_kj)
    return [[fold([a_k * row.weights[j] for a_k, row in zip(a, rows)]) for j in range(len(rows[0].weights))] for a in A]


def _columns(A: Matrix, start: int, width: int) -> Matrix:
    # columns start to start + width - 1 of A: a head's block of a packed matrix
    return [row[start : start + width] for row in A]


def _side_by_side(blocks: Sequence[Matrix]) -> Matrix:
    # _columns' inverse: the blocks' rows joined, block 0's first
    return [[value for block in blocks for value in block[t]] for t in range(len(blocks[0]))]


class AttentionLayer(ParameterFreeLayer[TokenNode]):
    """
    Over tokens tokens of features features (d), in heads heads (h) of key_size features (d_k,
    d / h when None): Q = X Wq^T + bq, K and V likewise, each head's P[i] =
    softmax_rows((Q[i] K[i]^T) / sqrt(d_k)) and H[i] = P[i] V[i], out = H Wo^T + bo with H the
    heads side by side. Its weight sets, also its draw and snapshot order, are Wq's h * d_k rows of
    d weights, then Wk's and Wv's, then Wo's d rows of h * d_k weights, each row with its bias,
    drawn as a dense layer's node, weights then bias: the weights decayed and the biases not.
    Hidden only, ending a token block's body.

    The backward pass runs whole in compute_hidden_deltas, since the gradients need dQ, dK and dV
    and the layer before reads dX.
    """

    example_fields: ClassVar[tuple[str, ...]] = ("_Q", "_K", "_V", "_P", "_H", "_dQ", "_dK", "_dV", "_dX")

    def __init__(
        self, input_layer: InputLayer, tokens: int, features: int, heads: int = 1, key_size: int | None = None
    ) -> None:
        assert len(input_layer.nodes) == tokens * features
        self.input_layer = input_layer
        self.tokens = tokens
        self.features = features
        self.heads = heads
        self.key_size = features // heads if key_size is None else key_size
        self.width = heads * self.key_size
        # computed once and divided by, never multiplied by its reciprocal
        self.scale = math.sqrt(self.key_size)
        self.queries, self.keys, self.values = ([WeightRow(features) for _ in range(self.width)] for _ in range(3))
        self.outputs = [WeightRow(self.width) for _ in range(features)]
        self.nodes = [TokenNode() for _ in range(tokens * features)]
        self.size = len(self.nodes)

        # this example's caches, set by the forward and backward passes
        self._Q: Matrix
        self._K: Matrix
        self._V: Matrix
        self._P: Matrix
        self._H: Matrix
        self._dQ: Matrix
        self._dK: Matrix
        self._dV: Matrix
        self._dX: list[float]

    def _inputs(self) -> Matrix:
        return [token_values(self.input_layer.nodes, t, self.features) for t in range(self.tokens)]

    def _deltas(self) -> Matrix:
        d = self.features
        return [[node.delta for node in self.nodes[t * d : (t + 1) * d]] for t in range(self.tokens)]

    def _head(self, A: Matrix, i: int) -> Matrix:
        return _columns(A, i * self.key_size, self.key_size)

    # project, attend, combine: the forward pass's blocks

    def _project(self, X: Matrix) -> tuple[Matrix, Matrix, Matrix]:
        # each over all heads, (T, h * d_k)
        return _affine(X, self.queries), _affine(X, self.keys), _affine(X, self.values)

    def _attend(self, Q: Matrix, K: Matrix, V: Matrix) -> tuple[Matrix, Matrix]:
        # each head's weights and mix: P (T, h * T) and H (T, h * d_k), packed
        weights: list[Matrix] = []
        mixes: list[Matrix] = []
        for i in range(self.heads):
            Qi, Ki, Vi = self._head(Q, i), self._head(K, i), self._head(V, i)
            P: Matrix = []
            for q in Qi:
                s = [_dot(q, k) / self.scale for k in Ki]
                m = max(s)
                e = [exp(s_u - m) for s_u in s]
                total = fold(e)
                P.append([e_u / total for e_u in e])
            weights.append(P)
            # H[i] = P[i] V[i]
            mixes.append([[_dot(p, v) for v in _transpose(Vi)] for p in P])
        return _side_by_side(weights), _side_by_side(mixes)

    def _combine(self, H: Matrix) -> Matrix:
        # one product over all heads' columns
        return _affine(H, self.outputs)

    def forward(self) -> None:
        self._Q, self._K, self._V = self._project(self._inputs())
        self._P, self._H = self._attend(self._Q, self._K, self._V)
        out = self._combine(self._H)
        for node, value in zip(self.nodes, (value for row in out for value in row)):
            node.activate(value)

    # the backward pass's blocks, last first

    def _combine_backward(self, delta: Matrix) -> Matrix:
        # dH = delta Wo, (T, h * d_k)
        return _times_rows(delta, self.outputs)

    def _attend_backward(self, dH: Matrix) -> tuple[Matrix, Matrix, Matrix]:
        # each head's dQ, dK and dV, packed (T, h * d_k)
        T = self.tokens
        dQs: list[Matrix] = []
        dKs: list[Matrix] = []
        dVs: list[Matrix] = []
        for i in range(self.heads):
            Q, K, V, dHi = self._head(self._Q, i), self._head(self._K, i), self._head(self._V, i), self._head(dH, i)
            P = _columns(self._P, i * T, T)
            # dP = dH V^T, dV = P^T dH
            dP = [[_dot(dh, v) for v in V] for dh in dHi]
            dVs.append([[_dot(p_column, dh_column) for dh_column in _transpose(dHi)] for p_column in _transpose(P)])
            dS: Matrix = []
            for p, dp in zip(P, dP):
                r = fold([dp_u * p_u for dp_u, p_u in zip(dp, p)])
                dS.append([p_u * (dp_u - r) for p_u, dp_u in zip(p, dp)])
            # dQ = (dS K) / s, dK = (dS^T Q) / s
            dQs.append([[_dot(ds, k_column) / self.scale for k_column in _transpose(K)] for ds in dS])
            dKs.append(
                [[_dot(ds_column, q_column) / self.scale for q_column in _transpose(Q)] for ds_column in _transpose(dS)]
            )
        return _side_by_side(dQs), _side_by_side(dKs), _side_by_side(dVs)

    def _project_backward(self, dQ: Matrix, dK: Matrix, dV: Matrix) -> list[float]:
        # three products over all heads, summed in this order
        dXq, dXk, dXv = _times_rows(dQ, self.queries), _times_rows(dK, self.keys), _times_rows(dV, self.values)
        return [(q + k) + v for q_row, k_row, v_row in zip(dXq, dXk, dXv) for q, k, v in zip(q_row, k_row, v_row)]

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
        for own_index, node in enumerate(self.nodes):
            node.delta = next_layer.downstream_sum(own_index)
        dH = self._combine_backward(self._deltas())
        self._dQ, self._dK, self._dV = self._attend_backward(dH)
        self._dX = self._project_backward(self._dQ, self._dK, self._dV)

    def downstream_sum(self, own_index: int) -> float:
        return self._dX[own_index]

    def accumulate_gradients(self) -> None:
        # each projection's rows over this example's tokens in order, after the examples before it
        X = self._inputs()
        for rows, gradient, inputs in (
            (self.queries, self._dQ, X),
            (self.keys, self._dK, X),
            (self.values, self._dV, X),
            (self.outputs, self._deltas(), self._H),
        ):
            for g_row, x in zip(gradient, inputs):
                for row, g in zip(rows, g_row):
                    accum = row.weight_gradient_accum
                    for j, x_j in enumerate(x):
                        accum[j] += g * x_j
                    row.bias_gradient_accum += g

    def weight_sets(self) -> Sequence[WeightRow]:
        return [*self.queries, *self.keys, *self.values, *self.outputs]

    def randomize_fan_in_aware(self, rng: Pcg64Generator) -> None:
        for row in self.weight_sets():
            row.randomize(rng)

    def snapshot_state(self) -> list[tuple[list[float], float]]:
        return snapshot_rows(self.weight_sets())

    def restore_state(self, layer_snapshot: Sequence[tuple[list[float], float]]) -> None:
        restore_rows(self.weight_sets(), layer_snapshot)
