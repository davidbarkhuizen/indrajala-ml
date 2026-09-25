from collections.abc import Sequence

from indrajala_ml.model.state_node import StateNode


class StateLayer:
    """
    The input layer: one StateNode per input dimension, set by update_state().
    """

    def __init__(self, dimension: int, bounds: Sequence[tuple[float, float]]) -> None:
        assert len(bounds) == dimension

        self.dimension: int = dimension
        self.nodes: Sequence[StateNode] = [StateNode() for _ in range(dimension)]

    def update_state(self, x_: tuple[float, ...]) -> None:
        assert len(x_) == self.dimension

        for i in range(self.dimension):
            self.nodes[i].update_value(x_[i])
