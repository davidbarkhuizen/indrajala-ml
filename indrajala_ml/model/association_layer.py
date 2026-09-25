from __future__ import annotations

from typing import Sequence

from indrajala_ml.model.association_node import AssociationNode
from indrajala_ml.model.state_layer import StateLayer


class AssociationLayer:
    """
    A layer of AssociationNodes, each fully connected to the input layer.
    """

    def __init__(self, size: int, input_layer: StateLayer | AssociationLayer) -> None:

        self.size: int = size

        self.input_layer: StateLayer | AssociationLayer = input_layer

        self.nodes: Sequence[AssociationNode] = [
            AssociationNode(input_nodes=self.input_layer.nodes) for _ in range(size)
        ]
