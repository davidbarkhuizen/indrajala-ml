"""
A network's layers as backend-free data (the composable-layers workplan, The design): a list of
layer specs, in forward order, that each implementation's builder maps to its own layer classes
(array_layer_builder.py for numpy and Rust, python_layer_builder.py for pure Python). Conv and Pool
are today's ConvSpec and PoolSpec, so the conv networks' conv_specs are already specs.

Activations are part of a Dense spec, not layers of their own: each stays fused into its layer, as
every Rust op is. validate_layer_specs accepts only the combinations every implementation builds,
so every accepted list is parity-testable.

Batch norm (the batch-norm workplan, D1) is a linear layer, then a BatchNorm that carries the
activation: Dense(30, activation="linear"), BatchNorm(activation="sigmoid"), or for conv
Conv(3, 8, activation="linear"), BatchNorm(activation="relu").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NoReturn

from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.window_geometry import output_size, pool_stride

Conv = ConvSpec
Pool = PoolSpec

Activation = Literal["sigmoid", "relu", "softmax", "linear"]
Loss = Literal["squared", "cross_entropy"]


@dataclass(frozen=True)
class Dense:
    """
    A fully connected layer of size nodes. A hidden layer is sigmoid or ReLU, and a sigmoid one
    may drop out (dropout: the drop probability, during training only). A linear hidden layer
    has no bias and no activation, and a BatchNorm follows it. The output layer
    (output=True, the last layer) is sigmoid, with the squared or the cross-entropy loss, or
    softmax, with the cross-entropy loss.
    """

    size: int
    activation: Activation = "sigmoid"
    dropout: float | None = None
    output: bool = False
    loss: Loss = "squared"


@dataclass(frozen=True)
class BatchNorm:
    """
    Batch normalization (Ioffe & Szegedy 2015; README, Batch normalization) of the linear layer
    before it, each feature over the batch (each channel over the batch and every position, after
    a conv layer), then the activation, sigmoid or ReLU (ReLU only after a conv layer). In training it
    normalizes with the batch's statistics and moves the running averages toward them at
    running_rate; in inference it normalizes with the running averages. epsilon is added to the
    variance. The defaults are PyTorch's.

    With a group_size (ghost batch norm, Hoffer et al. 2017; the batch-norm workplan, D6), a
    training batch is normalized in groups of group_size examples, in row order, each with its own
    statistics, and each group moves the running averages in turn. The last group is the
    remainder, and a batch that leaves it one example is refused (ghost_groups).
    """

    activation: Literal["sigmoid", "relu"] = "sigmoid"
    epsilon: float = 1e-5
    running_rate: float = 0.1
    group_size: int | None = None


LayerSpec = Dense | ConvSpec | PoolSpec | BatchNorm


def refuse_single_example(layer: object) -> NoReturn:
    """A batch-norm layer's, and its linear layer's, refusal of a one-example training step (the
    batch-norm workplan, D4), in every implementation."""
    raise ValueError(
        f"a {type(layer).__name__} trains on batches only: batch norm normalizes a batch of one to 0 "
        "(the batch-norm workplan, D4)"
    )


def refuse_single_example_network(specs: Sequence[LayerSpec], batch_norm_index: int) -> NoReturn:
    """A network's refusal of a one-example training step, naming its first batch-norm layer (D4)."""
    raise ValueError(
        f"layer {batch_norm_index}, {specs[batch_norm_index]!r}, can't train on one example: it "
        "would normalize every value to 0 and pass no gradient back. Train on batches of 2 or more (the "
        "batch-norm workplan, D4)"
    )


def ghost_groups(rows: int, group_size: int | None) -> list[tuple[int, int]]:
    """A training batch of rows examples as its ghost groups (D6), (first, end) example ranges: runs
    of group_size in row order, the last the remainder, or the whole batch without a group_size.
    Refuses a last group of one example, which would normalize to 0 (D4's reason)."""
    size = rows if group_size is None else group_size
    ranges = [(first, min(first + size, rows)) for first in range(0, rows, size)]
    if rows > 1 and ranges[-1][1] - ranges[-1][0] == 1:
        raise ValueError(_single_example_group(rows, size))
    return ranges


def _single_example_group(rows: int, group_size: int) -> str:
    return (
        f"a batch of {rows} in groups of {group_size} leaves a last group of one example, which batch norm "
        "normalizes to 0 (the batch-norm workplan, D6): use a batch size whose remainder isn't 1"
    )


def refuse_single_example_groups(specs: Sequence[LayerSpec], batch_size: int) -> None:
    """A network's refusal of a training batch of batch_size, 2 or more, that leaves some BatchNorm
    a last ghost group of one example (D6), naming the layer."""
    for i, spec in enumerate(specs):
        group_size = spec.group_size if isinstance(spec, BatchNorm) else None
        if group_size is not None and batch_size > group_size and batch_size % group_size == 1:
            raise ValueError(f"layer {i}, {spec!r}: {_single_example_group(batch_size, group_size)}")


def batch_norm_index(specs: Sequence[LayerSpec]) -> int | None:
    """The index of the first BatchNorm in specs, if any: such a network refuses a one-example
    training step (D4)."""
    return next((i for i, spec in enumerate(specs) if isinstance(spec, BatchNorm)), None)


# a network's input: (dimension,) for a flat input, or (height, width, channels) for an image, whose
# flat layout is channel-major (conv_layer.py, conv_array_layer.py)
InputShape = tuple[int] | tuple[int, int, int]
ImageShape = tuple[int, int, int]


def image_shape(shape: InputShape) -> ImageShape:
    """shape, which a conv or pool layer reads, as (height, width, channels)."""
    assert len(shape) == 3, f"a conv or pool layer needs a (height, width, channels) input; got {shape}"
    return shape


@dataclass(frozen=True)
class SpecShape:
    """
    One spec's place in its list's shape walk (spec_shapes): the shape it reads, the shape it gives,
    and for a BatchNorm its positions, 1 after a dense layer or a conv layer's out_height *
    out_width (1 for every other spec).
    """

    input_shape: InputShape
    output_shape: InputShape
    positions: int = 1


def spec_shapes(specs: Sequence[LayerSpec], input_shape: InputShape) -> list[SpecShape]:
    """
    Each of specs' shapes over input_shape, in forward order, each spec's input shape the previous
    one's output shape: what every builder needs besides the choice of class. A BatchNorm keeps its
    linear layer's shape (validate_layer_specs). The specs' own arguments are checked by the layers
    built from them, not here.
    """
    shapes: list[SpecShape] = []
    shape = input_shape
    for spec in specs:
        if isinstance(spec, Dense):
            shapes.append(SpecShape(shape, (spec.size,)))
        elif isinstance(spec, BatchNorm):
            positions = shape[0] * shape[1] if len(shape) == 3 else 1
            shapes.append(SpecShape(shape, shape, positions))
        else:
            height, width, channels = image_shape(shape)
            if isinstance(spec, ConvSpec):
                window, stride, channels = spec.kernel_size, spec.stride, spec.channel_count
            else:
                window, stride = spec.pool_size, pool_stride(spec.pool_size, spec.stride)
            output = (output_size(height, window, stride), output_size(width, window, stride), channels)
            shapes.append(SpecShape(shape, output))
        shape = shapes[-1].output_shape
    return shapes


def _check_batch_norm(spec: BatchNorm, before: LayerSpec | None) -> None:
    linear = isinstance(before, Dense | ConvSpec) and before.activation == "linear"
    assert linear, (
        f"a BatchNorm normalizes a linear layer, so one comes right before it; got {before!r} before {spec!r}"
    )
    if isinstance(before, ConvSpec):
        assert spec.activation == "relu", f"a BatchNorm after a conv layer is ReLU; got {spec!r}"
    assert spec.activation in ("sigmoid", "relu"), f"a BatchNorm is sigmoid or ReLU; got {spec!r}"
    assert spec.epsilon > 0.0, f"epsilon must be positive; got {spec!r}"
    assert 0.0 < spec.running_rate <= 1.0, f"running_rate must be in (0.0, 1.0]; got {spec!r}"
    assert spec.group_size is None or spec.group_size >= 2, f"group_size must be 2 or more; got {spec!r}"


def validate_layer_specs(specs: Sequence[LayerSpec]) -> None:
    """
    Rejects a spec list that some implementation can't build: a conv or pool layer after a dense
    one (a dense layer's fused hidden delta reads the next layer's W), a front end without a conv
    layer, a softmax hidden layer, dropout on anything but a sigmoid hidden layer (the dropout op
    is fused with the sigmoid), a linear layer without a BatchNorm right after it or a BatchNorm
    without one right before it (a conv one's BatchNorm is ReLU), and an output layer that isn't
    exactly the last layer.
    """
    assert specs, "a network needs at least one layer"
    *hidden, output = specs

    # the front end: every layer before the first dense one
    front_end_length = next((i for i, spec in enumerate(hidden) if isinstance(spec, Dense)), len(hidden))
    front_end = hidden[:front_end_length]
    assert not any(isinstance(spec, ConvSpec | PoolSpec) for spec in hidden[front_end_length:]), (
        f"conv and pool layers must all come before the dense layers; got {list(specs)!r}"
    )
    for i, spec in enumerate(front_end):
        after = hidden[i + 1] if i + 1 < len(hidden) else output
        if isinstance(spec, BatchNorm):
            _check_batch_norm(spec, front_end[i - 1] if i > 0 else None)
        elif isinstance(spec, ConvSpec):
            assert spec.activation in ("relu", "linear"), f"a conv layer is ReLU or linear; got {spec!r}"
            assert (spec.activation == "linear") == isinstance(after, BatchNorm), (
                f"a linear layer and a BatchNorm come as a pair, the linear layer first; got {spec!r} before {after!r}"
            )
    assert not front_end or any(isinstance(spec, ConvSpec) for spec in front_end), (
        "a front end of pool layers needs at least one conv layer"
    )

    dense = hidden[front_end_length:]
    for i, spec in enumerate(dense):
        after = dense[i + 1] if i + 1 < len(dense) else output
        if isinstance(spec, BatchNorm):
            _check_batch_norm(spec, dense[i - 1] if i > 0 else None)
            continue
        assert isinstance(spec, Dense), (
            f"conv and pool layers must all come before the dense layers; got {list(specs)!r}"
        )
        assert not spec.output, f"only the last layer is the output layer; got {spec!r} before it"
        assert spec.size >= 1, f"a dense layer needs at least one node; got {spec!r}"
        assert spec.activation in ("sigmoid", "relu", "linear"), (
            f"a hidden layer is sigmoid, ReLU or linear; got {spec!r}"
        )
        assert spec.dropout is None or spec.activation == "sigmoid", (
            f"dropout is fused with the sigmoid, so it needs a sigmoid hidden layer; got {spec!r}"
        )
        assert spec.loss == "squared", f"a loss belongs to the output layer; got {spec!r}"
        assert (spec.activation == "linear") == isinstance(after, BatchNorm), (
            f"a linear layer and a BatchNorm come as a pair, the linear layer first; got {spec!r} before {after!r}"
        )

    assert isinstance(output, Dense) and output.output, (
        f"the last layer must be the output layer, Dense(..., output=True); got {output!r}"
    )
    assert output.size >= 1, f"the output layer needs at least one node; got {output!r}"
    assert output.dropout is None, f"the output layer doesn't drop out; got {output!r}"
    assert output.activation in ("sigmoid", "softmax"), f"the output layer is sigmoid or softmax; got {output!r}"
    assert output.activation != "softmax" or output.loss == "cross_entropy", (
        f"a softmax output layer's delta is the cross-entropy loss's; got {output!r}"
    )
