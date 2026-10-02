# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, Q, K, V, P, H, which strict mode takes for constants)
"""
Single-head self-attention in pure Python (the layer-norm and attention workplan, stage 3; README,
Layer norm and attention), the counterpart of attention_array_layer.py: every expression the
README's, per scalar, in its grouping, and every sum, the products' included, a left fold from 0.0
in index order (fold).

The layer's per-example caches (Q, K, V, P, H, and backward dQ, dK, dV, dX) are its own, not its
nodes': they are matrices over the tokens. It names them in example_fields, and the layer-major
batch path (layer_major.py) keeps them in lanes, as it does the nodes' fields. Every pass assigns
new lists, never writes into a cached one, so a lane can hold them without a copy.

The softmax's exp is this module's exp, which tests may replace, as attention_array_layer's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from indrajala_ml.model.batch_norm_layer import fold
from indrajala_ml.model.layer_protocols import InputLayer, TrainableLayer
from indrajala_ml.model.residual_layer import ParameterFreeLayer
from indrajala_ml.model.token_layer import TokenNode, WeightRow, restore_rows, snapshot_rows, token_values
from indrajala_ml.pcg64 import Pcg64Generator

exp = math.exp

Matrix = list[list[float]]


def _project(X: Matrix, rows: Sequence[WeightRow]) -> Matrix:
    # X W^T + b: each token's product with each row, then the bias
    return [[fold([x_j * w_j for x_j, w_j in zip(x, row.weights)]) + row.bias for row in rows] for x in X]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return fold([a_i * b_i for a_i, b_i in zip(a, b)])


def _transpose(A: Matrix) -> Matrix:
    return [list(column) for column in zip(*A)]


def _times_rows(A: Matrix, rows: Sequence[WeightRow]) -> Matrix:
    # A W, W's rows the weight rows: (A W)_tj = sum_k(A_tk * W_kj)
    return [[fold([a_k * row.weights[j] for a_k, row in zip(a, rows)]) for j in range(len(rows[0].weights))] for a in A]


class AttentionLayer(ParameterFreeLayer[TokenNode]):
    """
    Q = X Wq^T + bq, K and V likewise, P = softmax_rows((Q K^T) / sqrt(d)), out = (P V) Wo^T + bo,
    over tokens tokens of features features. Its weight sets, also its draw and snapshot order, are
    Wq's rows, then Wk's, Wv's and Wo's, each row with its bias, drawn as a dense layer's node,
    weights then bias: the weights decayed and the biases not. Hidden only, ending a token block's
    body.

    The backward pass runs whole in compute_hidden_deltas, since the gradients need dQ, dK and dV
    and the layer before reads dX.
    """

    example_fields: ClassVar[tuple[str, ...]] = ("_Q", "_K", "_V", "_P", "_H", "_dQ", "_dK", "_dV", "_dX")

    def __init__(self, input_layer: InputLayer, tokens: int, features: int) -> None:
        assert len(input_layer.nodes) == tokens * features
        self.input_layer = input_layer
        self.tokens = tokens
        self.features = features
        self.scale = math.sqrt(features)
        self.queries, self.keys, self.values, self.outputs = (
            [WeightRow(features) for _ in range(features)] for _ in range(4)
        )
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

    def forward(self) -> None:
        X = self._inputs()
        self._Q = Q = _project(X, self.queries)
        self._K = K = _project(X, self.keys)
        self._V = V = _project(X, self.values)
        P: Matrix = []
        for q in Q:
            s = [_dot(q, k) / self.scale for k in K]
            m = max(s)
            e = [exp(s_u - m) for s_u in s]
            total = fold(e)
            P.append([e_u / total for e_u in e])
        self._P = P
        # H = P V
        self._H = H = [[_dot(p, v) for v in _transpose(V)] for p in P]
        out = _project(H, self.outputs)
        for node, value in zip(self.nodes, (value for row in out for value in row)):
            node.activate(value)

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
        for own_index, node in enumerate(self.nodes):
            node.delta = next_layer.downstream_sum(own_index)
        delta = self._deltas()
        P, V, Q, K = self._P, self._V, self._Q, self._K
        dH = _times_rows(delta, self.outputs)
        # dP = dH V^T, dV = P^T dH
        dP = [[_dot(dh, v) for v in V] for dh in dH]
        self._dV = [[_dot(p_column, dh_column) for dh_column in _transpose(dH)] for p_column in _transpose(P)]
        dS: Matrix = []
        for p, dp in zip(P, dP):
            r = fold([dp_u * p_u for dp_u, p_u in zip(dp, p)])
            dS.append([p_u * (dp_u - r) for p_u, dp_u in zip(p, dp)])
        # dQ = (dS K) / s, dK = (dS^T Q) / s
        self._dQ = [[_dot(ds, k_column) / self.scale for k_column in _transpose(K)] for ds in dS]
        self._dK = [
            [_dot(ds_column, q_column) / self.scale for q_column in _transpose(Q)] for ds_column in _transpose(dS)
        ]
        # three products, summed in this order
        dQ, dK, dV = (
            _times_rows(self._dQ, self.queries),
            _times_rows(self._dK, self.keys),
            _times_rows(self._dV, self.values),
        )
        self._dX = [
            (dq + dk) + dv for dq_row, dk_row, dv_row in zip(dQ, dK, dV) for dq, dk, dv in zip(dq_row, dk_row, dv_row)
        ]

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
