from collections.abc import Sequence

from indrajala_ml.model.base_node import AbstractNode, WeightedInputNode


class AssociationNode(WeightedInputNode):
    def __init__(
        self,
        input_nodes: Sequence[AbstractNode],
        input_node_weights: Sequence[float] | None = None,
        threshold: float = 0.0,
        default_input_node_weight: float = 1.0,
    ) -> None:
        super().__init__(
            input_nodes, input_node_weights, offset=threshold, default_input_node_weight=default_input_node_weight
        )

    @property
    def threshold(self) -> float:
        return self._offset

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._offset = value

    def value(self) -> float:
        return 1.0 if self.z() > 0.0 else 0.0

    def learn(self, learning_rate: float, reference_value: float) -> None:

        current_value = self.value()
        correctly_categorised = current_value == reference_value
        if correctly_categorised:
            return

        if reference_value == 1 and current_value == 0:
            d = 1.0
        elif reference_value == 0 and current_value == 1:
            d = -1.0
        else:
            raise ValueError(f"reference_value = {reference_value}, current_value = {current_value}")

        self.threshold = self.threshold + learning_rate * d * 1.0
        self.update_input_weights(
            [
                weight + learning_rate * d * node.value()
                for weight, node in zip(self.input_node_weights, self.input_nodes)
            ]
        )
