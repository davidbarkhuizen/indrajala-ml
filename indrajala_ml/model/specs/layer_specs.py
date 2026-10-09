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

A residual block (the residual-connections workplan, D1) is a nested spec, Residual(body), whose
output is its input plus its body's: x + F(x). The builders flatten it (expand_specs) into a Fork,
the body's layers, then an Add, so a network's layers are its expanded specs, not its specs.

A patch model (the layer-norm and attention workplan; README, Layer norm and attention) starts with
Patches, which cuts the image into tokens, has a token part of token-wise Dense layers, a Position
and residual blocks over tokens (LayerNorm, Attention), and ends it with TokenMean, before the
dense part. A token sequence of T tokens of d features is the shape (T, d), flat and token-major
(D2). LayerNorm stands in flat dense networks too (D5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec

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
    # an affine layer, W x + b with no activation, ends a residual block's body (the
    # residual-connections workplan, D4): bias=True on a linear layer, and nowhere else
    bias: bool = False


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


@dataclass(frozen=True)
class Residual:
    """
    out = x + body(x): the body's layers, then the block's input added to their output (He et al.
    2016's pre-activation form; README, Residual connections). The body is dense hidden layers, as
    a network's (a linear layer and a BatchNorm as a pair, dropout on sigmoid only), with no conv,
    pool or nested block (D2, D6), ending in an affine layer, Dense(n, "linear", bias=True), whose
    n is the block's input size (D5).
    """

    body: tuple[LayerSpec, ...]

    def __post_init__(self) -> None:
        # a list given as the body is kept as a tuple, so equal blocks compare equal (a loaded
        # file's against the network's, format2.check_loadable) and the spec stays hashable
        object.__setattr__(self, "body", tuple(self.body))


@dataclass(frozen=True)
class Patches:
    """
    An (H, W, C) image as (H/p * W/p, p * p * C) tokens, p the patch_size, which divides H and W:
    the patches in row-major order, each patch's values in the conv kernel's (channel, row, col)
    order (D3). Parameter-free, and the first layer.
    """

    patch_size: int


@dataclass(frozen=True)
class Position:
    """A learned (T, d) table added to the tokens, starting at zero (D7). Once, before any block."""


@dataclass(frozen=True)
class LayerNorm:
    """
    Each token's features (a flat layer's, as one token) normalized by their mean and biased
    variance, epsilon added to the variance, then gamma * xhat + beta per feature, gamma starting at
    1 and beta at 0 (D5). No activation after it, and the same in training and inference. It stands
    among the tokens, and wherever a dense hidden layer may.
    """

    epsilon: float = 1e-5


@dataclass(frozen=True)
class Attention:
    """
    Self-attention over the tokens in heads heads of key_size features each (d / heads when None,
    which heads must then divide), values as wide as keys, no mask, biases on all four
    projections, ending in the affine output projection (the layer-norm and attention workplan,
    D6; the multi-head attention workplan, D2, D3). It ends a token block's body.
    """

    heads: int = 1
    key_size: int | None = None

    def head_size(self, features: int) -> int:
        """Each head's width d_k over tokens of features features: key_size, else features / heads."""
        return features // self.heads if self.key_size is None else self.key_size


@dataclass(frozen=True)
class TokenMean:
    """The mean over the tokens, (T, d) to (d,) (D8): the token part's end."""


LayerSpec = Dense | ConvSpec | PoolSpec | BatchNorm | Residual | Patches | Position | LayerNorm | Attention | TokenMean

# the specs that act on tokens only, between Patches and TokenMean, both included
TokenSpec = Patches | Position | Attention | TokenMean


@dataclass(frozen=True)
class Fork:
    """
    A residual block's first layer once flattened (expand_specs), which no user writes: it passes
    its input on unchanged and keeps it for the block's Add.
    """


@dataclass(frozen=True)
class Add:
    """A residual block's last layer once flattened: the body's output plus the block's input."""


# a network's layers, one spec each: its specs with every Residual flattened
ExpandedSpec = (
    Dense | ConvSpec | PoolSpec | BatchNorm | Fork | Add | Patches | Position | LayerNorm | Attention | TokenMean
)


def expand_specs(specs: Sequence[LayerSpec | Fork | Add]) -> list[ExpandedSpec]:
    """
    specs with each Residual(body) flattened into Fork, *body, Add, in order: the one place blocks
    are flattened. Every layer index (snapshots, optimizer state, checkpoints, messages) is an
    index into this list. An already-expanded list is returned as it is.
    """
    expanded: list[ExpandedSpec] = []
    for spec in specs:
        if isinstance(spec, Residual):
            expanded += [Fork(), *expand_specs(spec.body), Add()]
        else:
            expanded.append(spec)
    return expanded


def spec_paths(specs: Sequence[LayerSpec]) -> list[str]:
    """
    Each expanded spec's place in specs as written, for messages: "layer 2", or inside a block
    "layer 2, block body 1", "layer 2, block fork" and "layer 2, block add".
    """
    paths: list[str] = []
    for i, spec in enumerate(specs):
        if isinstance(spec, Residual):
            body = [path.replace("layer", "block body", 1) for path in spec_paths(spec.body)]
            paths += [f"layer {i}, block fork", *(f"layer {i}, {path}" for path in body), f"layer {i}, block add"]
        else:
            paths.append(f"layer {i}")
    return paths
