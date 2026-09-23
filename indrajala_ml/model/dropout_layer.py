from __future__ import annotations

import random
from typing import Sequence

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode, sigmoid


def make_dropout_node_cls(drop_probability: float) -> type[BackpropNode]:
    """
    Returns a BackpropNode subclass whose forward() randomly zeroes its own activation with
    probability drop_probability, every training-time forward pass (Srivastava et al., 2014) -
    inactive at inference (self.training is False by default, and only ever set True for the
    duration of one learn()/learn_batch() call - see BackpropNetworkBase._set_training_mode).

    Inverted dropout: a kept unit's activation is rescaled by 1/keep_probability during
    training, so its expected contribution matches the full (all-units-kept) network - the
    standard convention specifically so no rescaling is needed at inference time, when every
    unit is kept unconditionally.

    A factory, not a fixed class, for the same reason as make_momentum_node_cls/
    make_l2_node_cls: there is no single drop_probability this codebase has measured and can
    recommend.
    """

    assert 0.0 <= drop_probability < 1.0, f"drop_probability must be in [0.0, 1.0); got {drop_probability}"
    keep_probability = 1.0 - drop_probability

    class DropoutNode(BackpropNode):
        def __init__(
            self,
            input_nodes: Sequence[BackpropNode],
            input_node_weights: Sequence[float] | None = None,
            bias: float = 0.0,
        ) -> None:
            super().__init__(input_nodes, input_node_weights, bias)
            # the safe default: a DropoutNode nobody ever calls set_training_mode(True) on
            # behaves exactly like a plain BackpropNode, never silently drops units
            self.training = False
            self._kept = True
            # pre-dropout-scaling sigmoid(z()), cached here rather than recomputed in
            # compute_hidden_delta - both because forward() already establishes that
            # cache-once convention (see BackpropNode.forward's own docstring) and because
            # the backward-pass derivative below needs the *unscaled* value, not self.value()
            self._base_activation = 0.0
            # a forward-time snapshot of self.training, NOT a live re-read of it in
            # compute_hidden_delta: BackpropClassifierNetwork.learn() only brackets
            # set_training_mode(True) around the forward() call itself (already back to False
            # by the time _backward() runs compute_hidden_delta), so the derivative's rescale
            # must be decided from what forward() actually did, not from whatever self.training
            # happens to read several calls later - the same reasoning _kept/_base_activation
            # are already forward-time snapshots for
            self._was_training = False

        def forward(self) -> float:
            self._base_activation = sigmoid(self.z())
            self._was_training = self.training
            if self.training:
                self._kept = random.random() >= drop_probability
                self._activation = (self._base_activation / keep_probability) if self._kept else 0.0
            else:
                self._kept = True
                self._activation = self._base_activation
            return self._activation

        def compute_output_delta(self, reference_value: float) -> None:
            raise NotImplementedError(
                "DropoutNode is a hidden-layer regularizer, not an output one - dropping units "
                "feeding the output layer isn't what this sibling builds."
            )

        def compute_hidden_delta(self, next_layer_nodes: Sequence["BackpropNode"], own_index: int) -> None:
            if not self._kept:
                # matches ReLUNode's own dead-unit precedent: a dropped unit contributed
                # nothing to this forward pass, so it gets zero delta, zero gradient, and its
                # incoming weights are untouched this step
                self.delta = 0.0
                return

            # self.value() (== self._activation) is the *post*-scaling activation
            # (_base_activation / keep_probability) - using it here the way
            # BackpropNode.compute_hidden_delta's own a*(1-a) does would be wrong, since
            # a*(1-a) != base*(1-base)/keep_probability in general. The correct chain rule for
            # d(base * mask/keep_probability)/dz is (mask/keep_probability) * base*(1-base) -
            # the ordinary sigmoid derivative on the *unscaled* activation, times the same
            # 1/keep_probability rescale forward() used.
            downstream = sum(node.delta * node.input_node_weights[own_index] for node in next_layer_nodes)
            sigmoid_derivative = self._base_activation * (1.0 - self._base_activation)
            scale = (1.0 / keep_probability) if self._was_training else 1.0
            self.delta = downstream * sigmoid_derivative * scale

    return DropoutNode


def make_dropout_layer_cls(drop_probability: float) -> type[BackpropLayer]:
    """
    The layer-level counterpart to make_dropout_node_cls - a BackpropLayer whose nodes are all
    DropoutNodes at the given drop_probability, plus the one real addition every other
    weight-update-rule sibling's own make_*_layer_cls doesn't need: set_training_mode, which
    propagates the train/eval flag down to each of this layer's nodes (see
    BackpropNetworkBase._set_training_mode, which calls this once per trainable layer).
    """

    class DropoutLayer(BackpropLayer):
        _node_cls = make_dropout_node_cls(drop_probability)

        def set_training_mode(self, training: bool) -> None:
            for node in self.nodes:
                node.training = training

    return DropoutLayer
