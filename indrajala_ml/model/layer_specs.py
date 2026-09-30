"""
A network's layers as backend-free data (the composable-layers workplan, The design): a list of
layer specs, in forward order, that each implementation's builder maps to its own layer classes
(array_layer_builder.py for numpy and Rust, python_layer_builder.py for pure Python). Conv and Pool
are today's ConvSpec and PoolSpec, so the conv networks' conv_specs are already specs.

Activations are part of a Dense spec, not layers of their own: each stays fused into its layer, as
every Rust op is. validate_layer_specs accepts only the combinations every implementation builds,
so every accepted list is parity-testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.max_pool_layer import PoolSpec

Conv = ConvSpec
Pool = PoolSpec

Activation = Literal["sigmoid", "relu", "softmax"]
Loss = Literal["squared", "cross_entropy"]


@dataclass(frozen=True)
class Dense:
    """
    A fully connected layer of size nodes. A hidden layer is sigmoid or ReLU, and a sigmoid one
    may drop out (dropout: the drop probability, during training only). The output layer
    (output=True, the last layer) is sigmoid, with the squared or the cross-entropy loss, or
    softmax, with the cross-entropy loss.
    """

    size: int
    activation: Activation = "sigmoid"
    dropout: float | None = None
    output: bool = False
    loss: Loss = "squared"


LayerSpec = Dense | ConvSpec | PoolSpec

# a network's input: (dimension,) for a flat input, or (height, width, channels) for an image, whose
# flat layout is channel-major (conv_layer.py, conv_array_layer.py)
InputShape = tuple[int] | tuple[int, int, int]


def validate_layer_specs(specs: Sequence[LayerSpec]) -> None:
    """
    Rejects a spec list that some implementation can't build: a conv or pool layer after a dense
    one (a dense layer's fused hidden delta reads the next layer's W), a front end without a conv
    layer, a softmax hidden layer, dropout on anything but a sigmoid hidden layer (the dropout op
    is fused with the sigmoid), and an output layer that isn't exactly the last layer.
    """
    assert specs, "a network needs at least one layer"
    *hidden, output = specs

    front_end = [spec for spec in hidden if isinstance(spec, ConvSpec | PoolSpec)]
    assert hidden[: len(front_end)] == front_end, (
        f"conv and pool layers must all come before the dense layers; got {list(specs)!r}"
    )
    assert not front_end or any(isinstance(spec, ConvSpec) for spec in front_end), (
        "a front end of pool layers needs at least one conv layer"
    )

    for spec in hidden[len(front_end) :]:
        assert isinstance(spec, Dense)
        assert not spec.output, f"only the last layer is the output layer; got {spec!r} before it"
        assert spec.size >= 1, f"a dense layer needs at least one node; got {spec!r}"
        assert spec.activation in ("sigmoid", "relu"), f"a hidden layer is sigmoid or ReLU; got {spec!r}"
        assert spec.dropout is None or spec.activation == "sigmoid", (
            f"dropout is fused with the sigmoid, so it needs a sigmoid hidden layer; got {spec!r}"
        )
        assert spec.loss == "squared", f"a loss belongs to the output layer; got {spec!r}"

    assert isinstance(output, Dense) and output.output, (
        f"the last layer must be the output layer, Dense(..., output=True); got {output!r}"
    )
    assert output.size >= 1, f"the output layer needs at least one node; got {output!r}"
    assert output.dropout is None, f"the output layer doesn't drop out; got {output!r}"
    assert output.activation in ("sigmoid", "softmax"), f"the output layer is sigmoid or softmax; got {output!r}"
    assert output.activation != "softmax" or output.loss == "cross_entropy", (
        f"a softmax output layer's delta is the cross-entropy loss's; got {output!r}"
    )
