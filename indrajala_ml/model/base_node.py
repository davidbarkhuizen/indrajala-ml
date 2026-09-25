from abc import ABC, abstractmethod
from collections.abc import Sequence


class AbstractNode(ABC):
    @abstractmethod
    def value(self) -> float:
        raise NotImplementedError()


class WeightedInputNode(AbstractNode):
    """
    What AssociationNode and BackpropNode share: weighted inputs plus an offset, summed by z(), and
    an updatable weight vector. value() (a hard step or a sigmoid) is the subclass's. The offset is
    stored as _offset and exposed by each subclass under its own name, threshold or bias.
    """

    def __init__(
        self,
        input_nodes: Sequence[AbstractNode],
        input_node_weights: Sequence[float] | None = None,
        offset: float = 0.0,
        default_input_node_weight: float = 1.0,
    ) -> None:

        self._offset: float = offset

        self.input_nodes: Sequence[AbstractNode] = input_nodes if input_nodes else []

        self.input_node_weights: Sequence[float] = (
            input_node_weights if input_node_weights else [default_input_node_weight for _ in self.input_nodes]
        )

    def update_input_weights(self, weights: list[float]) -> None:
        assert len(weights) == len(self.input_nodes)
        self.input_node_weights = weights

    def z(self) -> float:

        aggregate_input_value: float = sum(
            [self.input_nodes[i].value() * self.input_node_weights[i] for i in range(len(self.input_nodes))]
        )

        return aggregate_input_value + self._offset
