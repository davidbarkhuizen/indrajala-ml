from indrajala_ml.model.base_node import AbstractNode


class StateNode(AbstractNode):
    """
    One input value, as a node the first layer reads.
    """

    def __init__(self, value: float = 0.0) -> None:

        self.__value: float = value

    def value(self) -> float:
        return self.__value

    def update_value(self, value: float) -> None:
        self.__value = value
