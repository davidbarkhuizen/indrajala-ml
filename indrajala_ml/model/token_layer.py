"""
A patch model's token layers in pure Python (the layer-norm and attention workplan, stage 3; README,
Layer norm and attention), the counterparts of token_array_layer.py: Patches, Position, TokenMean
and the token-wise dense layer. A token sequence of T tokens of d features is T * d nodes,
token-major, node t * d + j (D2).

Every sum is a left fold from 0.0 in index order (fold), the token-wise dense layer's weighted
sums included: the README's order, which the builtin sum, compensated since Python 3.12, isn't. A
weight shared over the tokens (a token-wise dense layer's W and b, the position table) is a weight
set of its own (WeightRow, PositionRow), apart from the nodes, which hold each token's values.

Each node caches its value and delta per example (PassNode's example_fields), so the layer-major
batch path (layer_major.py) keeps them in lanes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Literal

from indrajala_ml.model.base_node import AbstractNode
from indrajala_ml.model.batch_norm_layer import fold
from indrajala_ml.model.fan_in_aware_init import fan_in_aware_weights_and_bias
from indrajala_ml.model.layer_protocols import InputLayer, TrainableLayer
from indrajala_ml.model.relu_layer import relu_activation, relu_delta
from indrajala_ml.model.residual_layer import ParameterFreeLayer, PassNode
from indrajala_ml.pcg64 import Pcg64Generator


class TokenNode(PassNode):
    """One feature of one token: its value and delta per example, computed by its layer."""

    def activate(self, value: float) -> None:
        self._activation = value


class WeightRow:
    """
    One row of a weight matrix shared over the tokens, with its bias: a token-wise dense layer's
    unit, or a row of an attention projection. The WeightSet the optimizer steps, decayed as a
    dense layer's node.
    """

    weights_decayed: ClassVar[bool] = True
    has_bias: ClassVar[bool] = True

    def __init__(self, fan_in: int) -> None:
        self._weights = [0.0] * fan_in
        self.bias = 0.0
        self.weight_gradient_accum = [0.0] * fan_in
        self.bias_gradient_accum = 0.0

    @property
    def weights(self) -> Sequence[float]:
        return self._weights

    def set_weights(self, weights: list[float]) -> None:
        assert len(weights) == len(self._weights)
        self._weights = weights

    def reset_gradient_accum(self) -> None:
        self.weight_gradient_accum = [0.0] * len(self._weights)
        self.bias_gradient_accum = 0.0

    def randomize(self, rng: Pcg64Generator) -> None:
        # a dense layer's node's draw: weights, then bias
        weights, self.bias = fan_in_aware_weights_and_bias(rng, len(self._weights))
        self.set_weights(weights)


def snapshot_rows(rows: Sequence[WeightRow]) -> list[tuple[list[float], float]]:
    """rows' (weights, bias), as a dense layer's snapshot_state."""
    return [(list(row.weights), row.bias) for row in rows]


def restore_rows(rows: Sequence[WeightRow], layer_snapshot: Sequence[tuple[list[float], float]]) -> None:
    for row, (weights, bias) in zip(rows, layer_snapshot, strict=True):
        row.set_weights(list(weights))
        row.bias = bias


def token_values(nodes: Sequence[AbstractNode], token: int, features: int) -> list[float]:
    """The values of token's features, nodes token-major."""
    return [node.value() for node in nodes[token * features : (token + 1) * features]]


class PatchesLayer(ParameterFreeLayer[TokenNode]):
    """
    An (H, W, C) image, channel-major (c * H * W + h * W + w), as T = H/p * W/p tokens of
    p * p * C features: token u * (W/p) + v is patch row u, column v, and its feature
    c * p * p + i * p + j is image[c, u * p + i, v * p + j] (D3). A fixed permutation; its
    downstream is the inverse permutation of its delta.
    """

    def __init__(self, input_layer: InputLayer, height: int, width: int, channels: int, patch_size: int) -> None:
        self.input_layer = input_layer
        p = patch_size
        rows, columns = height // p, width // p
        # source[i]: the image index node i reads
        self.source = [
            c * height * width + (u * p + i) * width + (v * p + j)
            for u in range(rows)
            for v in range(columns)
            for c in range(channels)
            for i in range(p)
            for j in range(p)
        ]
        self.target = [0] * len(self.source)
        for node_index, image_index in enumerate(self.source):
            self.target[image_index] = node_index
        self.nodes = [TokenNode() for _ in self.source]
        self.size = len(self.nodes)

    def forward(self) -> None:
        inputs = self.input_layer.nodes
        for node, image_index in zip(self.nodes, self.source):
            node.activate(inputs[image_index].value())

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
        for own_index, node in enumerate(self.nodes):
            node.delta = next_layer.downstream_sum(own_index)

    def downstream_sum(self, own_index: int) -> float:
        return self.nodes[self.target[own_index]].delta


class TokenMeanLayer(ParameterFreeLayer[TokenNode]):
    """
    The mean over the tokens, (T, d) to (d,) (D8): out_j = sum_t(x_tj) / T, a left fold over the
    tokens; dx_tj = delta_j / T for every t.
    """

    def __init__(self, input_layer: InputLayer, tokens: int, features: int) -> None:
        assert len(input_layer.nodes) == tokens * features
        self.input_layer = input_layer
        self.tokens = tokens
        self.nodes = [TokenNode() for _ in range(features)]
        self.size = features

    def forward(self) -> None:
        inputs, d = self.input_layer.nodes, self.size
        for j, node in enumerate(self.nodes):
            node.activate(fold([inputs[t * d + j].value() for t in range(self.tokens)]) / self.tokens)

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
        for own_index, node in enumerate(self.nodes):
            node.delta = next_layer.downstream_sum(own_index)

    def downstream_sum(self, own_index: int) -> float:
        return self.nodes[own_index % self.size].delta / self.tokens


class PositionRow:
    """One token's row of the position table: weights without a bias, never decayed (D7)."""

    weights_decayed: ClassVar[bool] = False
    has_bias: ClassVar[bool] = False

    def __init__(self, features: int) -> None:
        # starting at zero (D7)
        self._weights = [0.0] * features
        # the WeightSet surface's bias, which the optimizer never steps (has_bias)
        self.bias = 0.0
        self.weight_gradient_accum = [0.0] * features
        self.bias_gradient_accum = 0.0

    @property
    def weights(self) -> Sequence[float]:
        return self._weights

    def set_weights(self, weights: list[float]) -> None:
        assert len(weights) == len(self._weights)
        self._weights = weights

    def reset_gradient_accum(self) -> None:
        self.weight_gradient_accum = [0.0] * len(self._weights)
        self.bias_gradient_accum = 0.0


class PositionLayer(ParameterFreeLayer[TokenNode]):
    """
    A learned (T, d) table P added to the tokens (D7), starting at zero: out_t = x_t + P_t. Its
    delta is its downstream, and P's gradient is that delta summed over the examples. One weight
    set per token; nothing drawn.
    """

    def __init__(self, input_layer: InputLayer, tokens: int, features: int) -> None:
        assert len(input_layer.nodes) == tokens * features
        self.input_layer = input_layer
        self.features = features
        self.rows = [PositionRow(features) for _ in range(tokens)]
        self.nodes = [TokenNode() for _ in range(tokens * features)]
        self.size = len(self.nodes)

    def forward(self) -> None:
        d = self.features
        for i, (node, input_node) in enumerate(zip(self.nodes, self.input_layer.nodes)):
            node.activate(input_node.value() + self.rows[i // d].weights[i % d])

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
        for own_index, node in enumerate(self.nodes):
            node.delta = next_layer.downstream_sum(own_index)

    def accumulate_gradients(self) -> None:
        d = self.features
        for i, node in enumerate(self.nodes):
            self.rows[i // d].weight_gradient_accum[i % d] += node.delta

    def weight_sets(self) -> Sequence[PositionRow]:
        return self.rows

    def snapshot_state(self) -> list[tuple[list[float]]]:
        return [(list(row.weights),) for row in self.rows]

    def restore_state(self, layer_snapshot: Sequence[Sequence[list[float]]]) -> None:
        for row, (weights,) in zip(self.rows, layer_snapshot, strict=True):
            row.set_weights(list(weights))


class TokenDenseLayer(ParameterFreeLayer[TokenNode]):
    """
    A dense layer acting on each token (D4), its units (W's rows and b) shared over the tokens:
    z_tk = sum_j(x_tj * W_kj) + b_k, then ReLU or the identity. Its units are drawn as a dense
    layer's nodes, weights then bias, and stepped and decayed as theirs.
    """

    def __init__(self, input_layer: InputLayer, size: int, tokens: int, activation: Literal["relu", "linear"]) -> None:
        self.input_layer = input_layer
        self.tokens = tokens
        self.input_size = len(input_layer.nodes) // tokens
        assert self.input_size * tokens == len(input_layer.nodes)
        self.size = size
        self.activation = activation
        self.units = [WeightRow(self.input_size) for _ in range(size)]
        self.nodes = [TokenNode() for _ in range(tokens * size)]

    def forward(self) -> None:
        for t in range(self.tokens):
            x = token_values(self.input_layer.nodes, t, self.input_size)
            for k, unit in enumerate(self.units):
                z = fold([x_j * w_j for x_j, w_j in zip(x, unit.weights)]) + unit.bias
                self.nodes[t * self.size + k].activate(relu_activation(z) if self.activation == "relu" else z)

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
        for own_index, node in enumerate(self.nodes):
            downstream = next_layer.downstream_sum(own_index)
            node.delta = relu_delta(downstream, node.value()) if self.activation == "relu" else downstream

    def downstream_sum(self, own_index: int) -> float:
        # sum_k(delta_tk * W_kj), for node t * input_size + j before this layer
        t, j = divmod(own_index, self.input_size)
        deltas = self.nodes[t * self.size : (t + 1) * self.size]
        return fold([node.delta * unit.weights[j] for node, unit in zip(deltas, self.units)])

    def accumulate_gradients(self) -> None:
        # over this example's tokens in order, after the examples before it
        for t in range(self.tokens):
            x = token_values(self.input_layer.nodes, t, self.input_size)
            for k, unit in enumerate(self.units):
                delta = self.nodes[t * self.size + k].delta
                accum = unit.weight_gradient_accum
                for j, x_j in enumerate(x):
                    accum[j] += delta * x_j
                unit.bias_gradient_accum += delta

    def weight_sets(self) -> Sequence[WeightRow]:
        return self.units

    def randomize_fan_in_aware(self, rng: Pcg64Generator) -> None:
        for unit in self.units:
            unit.randomize(rng)

    def snapshot_state(self) -> list[tuple[list[float], float]]:
        return snapshot_rows(self.units)

    def restore_state(self, layer_snapshot: Sequence[tuple[list[float], float]]) -> None:
        restore_rows(self.units, layer_snapshot)
