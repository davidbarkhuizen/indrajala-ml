"""
Pure-Python batch normalization of a linear layer, fused with its activation (the batch-norm
workplan, D1 and D3), the counterpart of batch_norm_array_layer.py: a dense layer's each feature
over the batch, a conv layer's each channel over the batch and every position.
Every expression is the README's (Batch normalization), per scalar, in its grouping, and every sum
over the batch is a left fold from 0.0 in example order (fold), not the builtin sum, which adds
floats with compensated summation since Python 3.12.

A BatchNormNode holds its feature's or channel's parameters, and its lists and statistics over a
training batch, in the README's order: example by example, then position by position. After a dense
layer it is also the layer's node for its feature; after a conv layer each position has a
BatchNormPosition. The layer-major batch path (layer_major.py) drives the layer through
forward_batch, backward_batch and select_example; the example-major path only classifies through it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from indrajala_ml.model.backprop_node import sigmoid
from indrajala_ml.model.base_node import AbstractNode
from indrajala_ml.model.layer_protocols import InputLayer
from indrajala_ml.model.layer_specs import ghost_groups, refuse_single_example
from indrajala_ml.model.relu_layer import relu_activation
from indrajala_ml.pcg64 import Pcg64Generator


def fold(values: Sequence[float]) -> float:
    """The README's sum: a left fold from 0.0, in order. Layer norm's and attention's too."""
    total = 0.0
    for value in values:
        total += value
    return total


class GammaAsWeights:
    """
    A feature's gamma and beta as the WeightSet the optimizer steps: [gamma] as the weights and
    beta as the bias, never decayed (the batch-norm workplan's D7). Batch norm's BatchNormNode and layer norm's
    LayerNormFeature; each sets gamma, beta and the accumulators in its __init__.
    """

    weights_decayed: ClassVar[bool] = False
    has_bias: ClassVar[bool] = True

    gamma: float
    beta: float
    weight_gradient_accum: list[float]
    bias_gradient_accum: float

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


class BatchNormNode(GammaAsWeights, AbstractNode):
    """
    One feature or channel: gamma and beta (the WeightSet the optimizer steps, as [gamma] and a
    bias beta), the running averages, and over a training batch its lists of d = x - mu, x-hat, the
    activations, the deltas and dl/dx, with each ghost group's variance and std (one group
    without a group_size). As a dense layer's node, value(), delta and dx are the selected
    example's (BatchNormLayer.select_example), or the inference forward pass's.
    """

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
        self.vars: list[float] = []
        self.stds: list[float] = []

    @property
    def channel(self) -> BatchNormNode:
        # a dense layer's node is its own feature's parameters
        return self

    def value(self) -> float:
        return self._activation

    def activate(self, value: float) -> None:
        # the layer computes a feature's activation, over the batch or in inference
        self._activation = value


class BatchNormPosition(AbstractNode):
    """
    One position of a conv layer's channel: value(), delta and dx of the selected example
    (BatchNormLayer.select_example), or the inference forward pass's, from its channel's lists.
    """

    def __init__(self, input_node: AbstractNode, channel: BatchNormNode) -> None:
        self.input_node = input_node
        self.channel = channel

        # set by a forward pass or select_example; no default, as in BackpropNode
        self._activation: float
        self.delta: float
        self.dx: float

    def value(self) -> float:
        return self._activation

    def activate(self, value: float) -> None:
        self._activation = value


class BatchNormLayer:
    """
    y = gamma * xhat + beta, then the activation, per feature of the linear layer before it, xhat
    normalized with the batch's statistics in training and the running averages in inference. The
    running averages move in training forward passes only. Hidden only, and batch only in training
    (D4).

    training is set by set_training_mode. The backward pass reads _was_training, training as
    forward_batch saw it, as BatchNormArrayLayer's.

    After a conv layer, positions is its out_height * out_width: the input layer's nodes are
    size // positions channels, channel-major, each with its BatchNormNode in channels, and the
    layer's nodes are BatchNormPositions. After a dense layer, the nodes are the channels.

    With a group_size, a training batch's ghost groups (layer_specs.ghost_groups, D6) are each
    normalized as a batch of their own, and move the running averages in turn; the gradients of
    gamma and beta still sum over the whole batch.
    """

    def __init__(
        self,
        input_layer: InputLayer,
        activation: Literal["sigmoid", "relu"],
        epsilon: float,
        running_rate: float,
        positions: int = 1,
        group_size: int | None = None,
    ) -> None:
        self.input_layer = input_layer
        self.size = len(input_layer.nodes)
        assert positions >= 1 and self.size % positions == 0, (
            f"{self.size} values aren't {positions} positions per channel"
        )
        self.positions = positions
        self.activation = activation
        self.epsilon = epsilon
        self.running_rate = running_rate
        self.group_size = group_size
        self._groups: list[tuple[int, int]] = []
        self.training = False
        self._was_training = False
        self.nodes: list[BatchNormNode] | list[BatchNormPosition]
        if positions == 1:
            self.channels = self.nodes = [BatchNormNode(node) for node in input_layer.nodes]
        else:
            # a channel's parameters don't read one input node: its positions do
            self.channels = [BatchNormNode(input_layer.nodes[c * positions]) for c in range(self.size // positions)]
            self.nodes = [
                BatchNormPosition(node, self.channels[i // positions]) for i, node in enumerate(input_layer.nodes)
            ]

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
            channel = node.channel
            xhat = (node.input_node.value() - channel.running_mean) / math.sqrt(channel.running_var + epsilon)
            node.activate(self._activate(channel.gamma * xhat + channel.beta))

    def forward_batch(self, inputs: Sequence[Sequence[float]]) -> None:
        # inputs[e][i]: the linear layer's z for example e, value i (feature i, or channel
        # i // positions at position i % positions). A training forward pass: the layer-major path
        # runs only in learn_batch
        assert self.training, "forward_batch is the training forward pass"
        self._was_training = True
        if len(inputs) < 2:
            refuse_single_example(self)
        self._groups = ghost_groups(len(inputs), self.group_size)
        epsilon, rate, positions = self.epsilon, self.running_rate, self.positions
        for c, node in enumerate(self.channels):
            values = [row[c * positions + p] for row in inputs for p in range(positions)]
            node.d, node.xhat, node.activations, node.vars, node.stds = [], [], [], [], []
            for first, end in self._groups:
                x = values[first * positions : end * positions]
                m = len(x)
                mu = fold(x) / m
                d = [x_i - mu for x_i in x]
                ss = fold([d_i * d_i for d_i in d])
                var = ss / m
                std = math.sqrt(var + epsilon)
                xhat = [d_i / std for d_i in d]
                gamma, beta = node.gamma, node.beta
                node.activations += [self._activate(gamma * xhat_i + beta) for xhat_i in xhat]

                node.running_mean = (1 - rate) * node.running_mean + rate * mu
                node.running_var = (1 - rate) * node.running_var + rate * (ss / (m - 1))

                node.d += d
                node.xhat += xhat
                node.vars.append(var)
                node.stds.append(std)
            node.deltas, node.dxs = [], []

    def select_example(self, example: int) -> None:
        # the nodes' value(), delta and dx become example's, for the layers either side
        positions = self.positions
        for i, node in enumerate(self.nodes):
            channel = node.channel
            k = example * positions + i % positions
            node.activate(channel.activations[k])
            if channel.dxs:
                node.delta = channel.deltas[k]
                node.dx = channel.dxs[k]

    def compute_hidden_deltas(self, next_layer: Any) -> None:
        refuse_single_example(self)

    def backward_batch(self, downstream: Sequence[Sequence[float]]) -> None:
        # downstream[e][i], the next layer's downstream sum: dl/dy is it times the activation's
        # derivative, as BatchNormArrayLayer.compute_hidden_delta_batch's; then dl/dx by the paper's
        # § 3 chain rule, term by term
        assert self._was_training, "the backward pass needs a training forward pass's batch statistics"
        epsilon, positions = self.epsilon, self.positions
        for c, node in enumerate(self.channels):
            ds = [row[c * positions + p] for row in downstream for p in range(positions)]
            if self.activation == "sigmoid":
                deltas = [ds_i * a * (1.0 - a) for ds_i, a in zip(ds, node.activations)]
            else:
                deltas = [ds_i * (1.0 if a > 0.0 else 0.0) for ds_i, a in zip(ds, node.activations)]
            gamma = node.gamma
            dxs: list[float] = []
            for (first, end), var, std in zip(self._groups, node.vars, node.stds):
                d = node.d[first * positions : end * positions]
                m = len(d)
                dxhat = [delta_i * gamma for delta_i in deltas[first * positions : end * positions]]
                inv_std = 1 / std
                inv_std3 = inv_std / (var + epsilon)
                dvar = fold([dxhat_i * d_i * -0.5 * inv_std3 for dxhat_i, d_i in zip(dxhat, d)])
                dmu = fold([dxhat_i * -inv_std for dxhat_i in dxhat]) + dvar * fold([-2 * d_i for d_i in d]) / m
                dxs += [dxhat_i * inv_std + dvar * (2 * d_i) / m + dmu / m for dxhat_i, d_i in zip(dxhat, d)]
            node.deltas = deltas
            node.dxs = dxs

    def downstream_sum(self, own_index: int) -> float:
        # dl/dx of the selected example: the linear layer's delta
        return self.nodes[own_index].dx

    def accumulate_gradients(self) -> None:
        # over the whole batch at once
        for node in self.channels:
            node.weight_gradient_accum[0] += fold([delta_i * xhat_i for delta_i, xhat_i in zip(node.deltas, node.xhat)])
            node.bias_gradient_accum += fold(node.deltas)

    def weight_sets(self) -> Sequence[BatchNormNode]:
        return self.channels

    def randomize_fan_in_aware(self, rng: Pcg64Generator) -> None:
        # D5: nothing drawn, so no other layer's draws shift
        pass

    def snapshot_state(self) -> list[tuple[list[float], float, float, float]]:
        return [([node.gamma], node.beta, node.running_mean, node.running_var) for node in self.channels]

    def restore_state(self, layer_snapshot: Sequence[Sequence[Any]]) -> None:
        for node, (weights, beta, running_mean, running_var) in zip(self.channels, layer_snapshot):
            (node.gamma,) = weights
            node.beta, node.running_mean, node.running_var = beta, running_mean, running_var
