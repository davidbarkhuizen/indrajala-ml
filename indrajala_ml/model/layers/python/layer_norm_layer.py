"""
Layer norm in pure Python (the layer-norm and attention workplan, stage 3; README, Layer norm and
attention), the counterpart of layer_norm_array_layer.py: each token's features, or a flat layer's
as one token, normalized by their own mean and biased variance, then gamma * xhat + beta. No batch
statistics, so the same in training and inference, and learn trains it as learn_batch does. Every
expression is the README's, per scalar, in its grouping, and every sum a left fold from 0.0 (fold).

A LayerNormFeature holds one feature's gamma and beta, shared over the tokens; a LayerNormNode one
token's feature, with its value, delta, xhat, its token's std and dl/dx per example (example_fields),
which the layer-major batch path (layer_major.py) keeps in lanes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

from indrajala_ml.model.layers.python.batch_norm_layer import GammaAsWeights, fold
from indrajala_ml.model.layers.python.residual_layer import ParameterFreeLayer, PassNode
from indrajala_ml.model.layers.python.token_layer import token_values
from indrajala_ml.model.protocols.layer_protocols import InputLayer, TrainableLayer


class LayerNormFeature(GammaAsWeights):
    """
    One feature's gamma and beta: the WeightSet the optimizer steps, as [gamma] and a bias beta,
    never decayed, as batch norm's (GammaAsWeights). Nothing drawn: 1 and 0.
    """

    def __init__(self) -> None:
        self.gamma = 1.0
        self.beta = 0.0
        self.weight_gradient_accum = [0.0]
        self.bias_gradient_accum = 0.0


class LayerNormNode(PassNode):
    """One token's feature: its value and delta, and its xhat, std and dl/dx, per example."""

    example_fields: ClassVar[tuple[str, ...]] = ("_activation", "delta", "xhat", "std", "dx")

    def __init__(self) -> None:
        super().__init__()
        # set by the forward and backward passes; no default, as in BackpropNode
        self.xhat: float
        self.std: float
        self.dx: float

    def activate(self, value: float) -> None:
        self._activation = value


class LayerNormLayer(ParameterFreeLayer[LayerNormNode]):
    """
    y = gamma * xhat + beta over each of tokens tokens of features features, gamma and beta shared
    over the tokens. A flat layer is one token of its whole size, a conv front end's image
    included. Hidden only.
    """

    def __init__(self, input_layer: InputLayer, tokens: int, features: int, epsilon: float) -> None:
        assert len(input_layer.nodes) == tokens * features
        self.input_layer = input_layer
        self.tokens = tokens
        self.features = features
        self.epsilon = epsilon
        self.channels = [LayerNormFeature() for _ in range(features)]
        self.nodes = [LayerNormNode() for _ in range(tokens * features)]
        self.size = len(self.nodes)

    def _token(self, t: int) -> Sequence[LayerNormNode]:
        return self.nodes[t * self.features : (t + 1) * self.features]

    def forward(self) -> None:
        d = self.features
        for t in range(self.tokens):
            x = token_values(self.input_layer.nodes, t, d)
            mu = fold(x) / d
            c = [x_j - mu for x_j in x]
            var = fold([c_j * c_j for c_j in c]) / d
            std = math.sqrt(var + self.epsilon)
            for node, c_j, channel in zip(self._token(t), c, self.channels):
                node.xhat = c_j / std
                node.std = std
                node.activate(channel.gamma * node.xhat + channel.beta)

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
        # the delta, the next layer's downstream, then each token's dl/dx, which needs all of the
        # token's deltas
        for own_index, node in enumerate(self.nodes):
            node.delta = next_layer.downstream_sum(own_index)
        d = self.features
        for t in range(self.tokens):
            nodes = self._token(t)
            dxhat = [node.delta * channel.gamma for node, channel in zip(nodes, self.channels)]
            m1 = fold(dxhat) / d
            m2 = fold([dxhat_j * node.xhat for dxhat_j, node in zip(dxhat, nodes)]) / d
            for node, dxhat_j in zip(nodes, dxhat):
                node.dx = ((dxhat_j - m1) - node.xhat * m2) / node.std

    def downstream_sum(self, own_index: int) -> float:
        return self.nodes[own_index].dx

    def accumulate_gradients(self) -> None:
        # over this example's tokens in order, after the examples before it
        d = self.features
        for i, node in enumerate(self.nodes):
            channel = self.channels[i % d]
            channel.weight_gradient_accum[0] += node.delta * node.xhat
            channel.bias_gradient_accum += node.delta

    def weight_sets(self) -> Sequence[LayerNormFeature]:
        return self.channels

    def snapshot_state(self) -> list[tuple[list[float], float]]:
        return [([channel.gamma], channel.beta) for channel in self.channels]

    def restore_state(self, layer_snapshot: Sequence[Sequence[Any]]) -> None:
        for channel, (weights, beta) in zip(self.channels, layer_snapshot, strict=True):
            (channel.gamma,) = weights
            channel.beta = beta
