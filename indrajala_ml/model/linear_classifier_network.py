from __future__ import annotations

import random

from indrajala_ml.model.association_layer import AssociationLayer
from indrajala_ml.model.bounds import half_widths as _half_widths
from indrajala_ml.model.bounds import validate_input_bounds
from indrajala_ml.model.state_layer import StateLayer


class LinearClassifierNetwork:
    def __init__(
        self,
        cardinality: int,
        dimension: int,
        input_bounds: list[tuple[float, float]],
        required_active: int | None = None,
    ) -> None:

        assert cardinality >= 1
        self.cardinality = cardinality

        self.dimension = dimension

        validate_input_bounds(dimension, input_bounds)
        self.input_bounds = input_bounds

        # how many hidden nodes must be active for the output to fire: cardinality (the
        # default) is AND, 1 is OR, anything between a k-of-n gate, as in a MADALINE committee
        required_active = cardinality if required_active is None else required_active
        assert 1 <= required_active <= cardinality
        self.required_active = required_active

        self.input_layer = StateLayer(dimension, input_bounds)

        self.hidden_layer = AssociationLayer(size=cardinality, input_layer=self.input_layer)

        # the output layer is a single neuron, fully connected to the hidden layer, that
        # activates once at least required_active of its input nodes are active
        self.output_layer = AssociationLayer(size=1, input_layer=self.hidden_layer)
        output_node = self.output_layer.nodes[0]
        output_node.update_input_weights([1.0 for _ in self.hidden_layer.nodes])
        output_node.threshold = -float(self.required_active - 1)

    def update_state_layer(self, x_: tuple[float, ...]) -> None:
        self.input_layer.update_state(x_)

    def classify_state(self, state: tuple[float, ...]) -> float:
        self.update_state_layer(state)
        return self.output_layer.nodes[0].value()

    def learn(self, learning_rate: float, state: tuple[float, ...], category: float) -> None:

        self.update_state_layer(state)

        output = self.output_layer.nodes[0].value()
        if output == category:
            return

        if category == 1:
            # false negative: too few hidden nodes are active - flipping any inactive one
            # to active can only increase the count, moving the output toward firing.
            candidates = [node for node in self.hidden_layer.nodes if node.value() == 0.0]
        else:
            # false positive: flipping an active node to inactive can only move the output
            # toward silent. Both branches hold for any required_active: the output is a
            # non-decreasing function of how many hidden nodes are active.
            candidates = [node for node in self.hidden_layer.nodes if node.value() == 1.0]

        responsible_node = min(candidates, key=lambda node: abs(node.z()))
        responsible_node.learn(learning_rate, category)

    def half_widths(self) -> list[float]:
        return _half_widths(self.input_bounds)

    def randomize(self) -> None:
        # each weight's range scales inversely with its dimension's half-width, so w_i * x_i
        # has a similar magnitude whatever that dimension's bounds; the threshold's range then
        # needn't scale. With fixed ranges the threshold dominated w.x at small bounds, making
        # almost every random node permanently on or off (at half-width 0.001, 0 of 200 random
        # cardinality=1 classifiers could reach both classes). At half-width 10, which the
        # demos and tests use, this is uniform(-2, 2).
        half_widths = self.half_widths()
        for node in self.hidden_layer.nodes:
            node.update_input_weights(
                [random.uniform(-20.0 / half_width, 20.0 / half_width) for half_width in half_widths]
            )
            node.threshold = random.uniform(-5, 5)

    @classmethod
    def randomized(
        cls,
        cardinality: int,
        dimension: int,
        input_bounds: list[tuple[float, float]],
        required_active: int | None = None,
    ) -> LinearClassifierNetwork:
        network = cls(cardinality, dimension, input_bounds, required_active)
        network.randomize()
        return network

    def snapshot(self) -> list[tuple[list[float], float]]:
        return [(list(node.input_node_weights), node.threshold) for node in self.hidden_layer.nodes]

    def restore(self, snapshot: list[tuple[list[float], float]]) -> None:
        for node, (weights, threshold) in zip(self.hidden_layer.nodes, snapshot):
            node.update_input_weights(weights)
            node.threshold = threshold
