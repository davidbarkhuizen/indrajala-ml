from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from indrajala_ml.model.layers.python.backprop_layer import BackpropLayer
from indrajala_ml.model.layers.python.backprop_node import BackpropNode, sigmoid
from indrajala_ml.pcg64 import Pcg64Generator, default_rng


class TrainingModeNode(BackpropNode):
    """A node whose forward pass differs in training: the base of make_dropout_node_cls's nodes."""

    training: bool
    rng: Pcg64Generator


def make_dropout_node_cls(drop_probability: float) -> type[TrainingModeNode]:
    """
    A BackpropNode subclass that zeroes its activation with probability drop_probability on each
    training forward pass (Srivastava et al., 2014), and does nothing at inference. self.training is
    False by default and set only for one learn()/learn_batch() call
    (BackpropNetworkBase._set_training_mode).

    Inverted dropout: a kept activation is scaled by 1/keep_probability in training, so its
    expected contribution matches the full network and inference needs no rescaling. The keep
    draw comes from rng, the network's generator, which DropoutLayer.set_rng hands to every node;
    a node on its own draws from OS entropy.
    """

    assert 0.0 <= drop_probability < 1.0, f"drop_probability must be in [0.0, 1.0); got {drop_probability}"
    keep_probability = 1.0 - drop_probability

    class DropoutNode(TrainingModeNode):
        # the mask and the unscaled activation are per example too (layer_major.py)
        example_fields: ClassVar[tuple[str, ...]] = (
            "_activation",
            "delta",
            "_kept",
            "_base_activation",
            "_was_training",
        )

        def __init__(
            self,
            input_nodes: Sequence[BackpropNode],
            input_node_weights: Sequence[float] | None = None,
            bias: float = 0.0,
        ) -> None:
            super().__init__(input_nodes, input_node_weights, bias)
            # off by default: a node never set to training behaves as a plain BackpropNode
            self.training = False
            self._kept = True
            # sigmoid(z()) before the dropout scaling, which the backward pass needs
            self._base_activation = 0.0
            # training as forward() saw it: learn() switches training off before the backward
            # pass, so compute_hidden_delta must not read self.training
            self._was_training = False
            self.rng: Pcg64Generator = default_rng()

        def forward(self) -> float:
            self._base_activation = sigmoid(self.z())
            self._was_training = self.training
            if self.training:
                self._kept = self.rng.random() >= drop_probability
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

        def compute_hidden_delta(self, downstream: float) -> None:
            if not self._kept:
                # a dropped unit contributed nothing, so it gets no delta and no update
                self.delta = 0.0
                return

            # d(base * mask/keep_probability)/dz = (mask/keep_probability) * base*(1-base): the
            # sigmoid derivative of the unscaled activation, not of self.value()
            sigmoid_derivative = self._base_activation * (1.0 - self._base_activation)
            scale = (1.0 / keep_probability) if self._was_training else 1.0
            self.delta = downstream * sigmoid_derivative * scale

    return DropoutNode


def make_dropout_layer_cls(drop_probability: float) -> type[BackpropLayer]:
    """
    A BackpropLayer of make_dropout_node_cls nodes, whose set_training_mode passes the flag to its
    nodes, and set_rng the network's generator.
    """

    class DropoutLayer(BackpropLayer):
        _node_cls = make_dropout_node_cls(drop_probability)

        def set_training_mode(self, training: bool) -> None:
            for node in self.nodes:
                assert isinstance(node, TrainingModeNode)
                node.training = training

        def set_rng(self, rng: Pcg64Generator) -> None:
            for node in self.nodes:
                assert isinstance(node, TrainingModeNode)
                node.rng = rng

    return DropoutLayer
