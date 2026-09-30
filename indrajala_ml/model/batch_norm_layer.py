"""
Pure-Python batch normalization of a dense linear layer, each feature over the batch, fused with
its activation (the batch-norm workplan, D1 and D3), the counterpart of batch_norm_array_layer.py.
Every expression is the README's (Batch normalization), per scalar, in its grouping, and every sum
over the batch is a left fold from 0.0 in example order (_fold), not the builtin sum, which adds
floats with compensated summation since Python 3.12.

A BatchNormNode holds its feature's per-example lists and statistics over a training batch. The
layer-major batch path (layer_major.py) drives the layer through forward_batch, backward_batch and
select_example; the example-major path only classifies through it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from indrajala_ml.model.backprop_node import sigmoid
from indrajala_ml.model.base_node import AbstractNode
from indrajala_ml.model.layer_protocols import InputLayer
from indrajala_ml.model.layer_specs import refuse_single_example
from indrajala_ml.model.relu_layer import relu_activation


def _fold(values: Sequence[float]) -> float:
    # the README's sum: a left fold from 0.0, in order
    total = 0.0
    for value in values:
        total += value
    return total


class BatchNormNode(AbstractNode):
    """
    One feature: gamma and beta (the WeightSet the optimizer steps, as [gamma] and a bias beta),
    the running averages, and over a training batch its lists of d = x - mu, x-hat, the activations,
    the deltas and dl/dx, with the batch's variance and std. value(), delta and dx are the selected
    example's (BatchNormLayer.select_example), or the inference forward pass's.
    """

    # gamma and beta are not decayed (D7); beta steps as a bias
    weights_decayed: ClassVar[bool] = False
    has_bias: ClassVar[bool] = True

    def __init__(self, input_node: AbstractNode) -> None:
        self.input_node = input_node

        # D5's initialization: nothing drawn
        self.gamma = 1.0
        self.beta = 0.0
        self.running_mean = 0.0
        self.running_var = 1.0

        self.weight_gradient_accum: list[float] = [0.0]
        self.bias_gradient_accum: float = 0.0

        # set by a forward pass, select_example or the backward pass; no default, as in BackpropNode
        self._activation: float
        self.delta: float
        self.dx: float

        # over a training batch (BatchNormLayer.forward_batch, backward_batch)
        self.d: list[float] = []
        self.xhat: list[float] = []
        self.activations: list[float] = []
        self.deltas: list[float] = []
        self.dxs: list[float] = []
        self.var = 0.0
        self.std = 0.0

    def value(self) -> float:
        return self._activation

    def activate(self, value: float) -> None:
        # the layer computes a feature's activation, over the batch or in inference
        self._activation = value

    # the WeightSet surface (layer_protocols.py)
    @property
    def weights(self) -> Sequence[float]:
        return [self.gamma]

    def set_weights(self, weights: list[float]) -> None:
        (self.gamma,) = weights

    @property
    def bias(self) -> float:
        return self.beta

    @bias.setter
    def bias(self, value: float) -> None:
        self.beta = value

    def reset_gradient_accum(self) -> None:
        self.weight_gradient_accum = [0.0]
        self.bias_gradient_accum = 0.0


class BatchNormLayer:
    """
    y = gamma * xhat + beta, then the activation, per feature of the linear layer before it, xhat
    normalized with the batch's statistics in training and the running averages in inference. The
    running averages move in training forward passes only. Hidden only, and batch only in training
    (D4).

    training is set by set_training_mode. The backward pass reads _was_training, training as
    forward_batch saw it, as BatchNormArrayLayer's.
    """

    def __init__(
        self, input_layer: InputLayer, activation: Literal["sigmoid", "relu"], epsilon: float, running_rate: float
    ) -> None:
        self.input_layer = input_layer
        self.size = len(input_layer.nodes)
        self.activation = activation
        self.epsilon = epsilon
        self.running_rate = running_rate
        self.training = False
        self._was_training = False
        self.nodes: list[BatchNormNode] = [BatchNormNode(node) for node in input_layer.nodes]

    def _activate(self, y: float) -> float:
        return sigmoid(y) if self.activation == "sigmoid" else relu_activation(y)

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def forward(self) -> None:
        # classify_state's single-example forward pass: inference only
        if self.training:
            refuse_single_example(self)
        epsilon = self.epsilon
        for node in self.nodes:
            xhat = (node.input_node.value() - node.running_mean) / math.sqrt(node.running_var + epsilon)
            node.activate(self._activate(node.gamma * xhat + node.beta))

    def forward_batch(self, inputs: Sequence[Sequence[float]]) -> None:
        # inputs[e][j]: the linear layer's z for example e, feature j. A training forward pass: the
        # layer-major path runs only in learn_batch
        assert self.training, "forward_batch is the training forward pass"
        self._was_training = True
        m = len(inputs)
        if m < 2:
            refuse_single_example(self)
        epsilon, rate = self.epsilon, self.running_rate
        for j, node in enumerate(self.nodes):
            x = [row[j] for row in inputs]
            mu = _fold(x) / m
            d = [x_i - mu for x_i in x]
            ss = _fold([d_i * d_i for d_i in d])
            var = ss / m
            std = math.sqrt(var + epsilon)
            xhat = [d_i / std for d_i in d]
            gamma, beta = node.gamma, node.beta
            node.activations = [self._activate(gamma * xhat_i + beta) for xhat_i in xhat]

            node.running_mean = (1 - rate) * node.running_mean + rate * mu
            node.running_var = (1 - rate) * node.running_var + rate * (ss / (m - 1))

            node.d, node.xhat, node.var, node.std = d, xhat, var, std
            node.deltas, node.dxs = [], []

    def select_example(self, example: int) -> None:
        # the nodes' value(), delta and dx become example's, for the layers either side
        for node in self.nodes:
            node.activate(node.activations[example])
            if node.dxs:
                node.delta = node.deltas[example]
                node.dx = node.dxs[example]

    def compute_hidden_deltas(self, next_layer: Any) -> None:
        refuse_single_example(self)

    def backward_batch(self, downstream: Sequence[Sequence[float]]) -> None:
        # downstream[e][j], the next layer's downstream sum: dl/dy is it times the activation's
        # derivative, as BatchNormArrayLayer.compute_hidden_delta_batch's; then dl/dx by the paper's
        # § 3 chain rule, term by term
        assert self._was_training, "the backward pass needs a training forward pass's batch statistics"
        epsilon = self.epsilon
        for j, node in enumerate(self.nodes):
            m = len(node.d)
            if self.activation == "sigmoid":
                deltas = [row[j] * a * (1.0 - a) for row, a in zip(downstream, node.activations)]
            else:
                deltas = [row[j] * (1.0 if a > 0.0 else 0.0) for row, a in zip(downstream, node.activations)]
            gamma, d = node.gamma, node.d
            dxhat = [delta_i * gamma for delta_i in deltas]
            inv_std = 1 / node.std
            inv_std3 = inv_std / (node.var + epsilon)
            dvar = _fold([dxhat_i * d_i * -0.5 * inv_std3 for dxhat_i, d_i in zip(dxhat, d)])
            dmu = _fold([dxhat_i * -inv_std for dxhat_i in dxhat]) + dvar * _fold([-2 * d_i for d_i in d]) / m
            node.deltas = deltas
            node.dxs = [dxhat_i * inv_std + dvar * (2 * d_i) / m + dmu / m for dxhat_i, d_i in zip(dxhat, d)]

    def downstream_sum(self, own_index: int) -> float:
        # dl/dx of the selected example: the linear layer's delta
        return self.nodes[own_index].dx

    def accumulate_gradients(self) -> None:
        # over the whole batch at once
        for node in self.nodes:
            node.weight_gradient_accum[0] += _fold(
                [delta_i * xhat_i for delta_i, xhat_i in zip(node.deltas, node.xhat)]
            )
            node.bias_gradient_accum += _fold(node.deltas)

    def weight_sets(self) -> Sequence[BatchNormNode]:
        return self.nodes

    def randomize_fan_in_aware(self) -> None:
        # D5: nothing drawn, so no other layer's draws shift
        pass

    def snapshot_state(self) -> list[tuple[list[float], float, float, float]]:
        return [([node.gamma], node.beta, node.running_mean, node.running_var) for node in self.nodes]

    def restore_state(self, layer_snapshot: Sequence[Sequence[Any]]) -> None:
        for node, (weights, beta, running_mean, running_var) in zip(self.nodes, layer_snapshot):
            (node.gamma,) = weights
            node.beta, node.running_mean, node.running_var = beta, running_mean, running_var
