"""
The window-recurrent network's windows (indrajala_ml/model/networks/window_recurrent_array_network.py),
as layer specs over a window's tokens and its output classes: W11's transformer window and the
shapes built from it (branched, recombining, an answers' branch), E6's linear baseline and its
FFN-only control. The window-recurrent studies train them (indrajala-studies); they live here so a
network can be built, tested and timed from the library alone.
"""

from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Branches,
    Dense,
    Embedding,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
    StateInput,
)

D, BLOCKS, HEADS, FFN = 16, 3, 2, 32  # W11: the stream's width, the blocks, the heads, the FFN's width


def transformer_window(tokens: int, classes: int, width: int = D) -> list[LayerSpec]:
    """
    W11's window over tokens tokens, classes output classes (a task's, or the modular family's); width
    its residual stream's, D by default, its FFN FFN / D times it (the window branches workplan's W1).
    """
    blocks: list[LayerSpec] = []
    for _ in range(BLOCKS):
        blocks += [
            StateInput(),
            Residual((LayerNorm(), Attention(heads=HEADS, causal=True))),
            Residual(
                (LayerNorm(), Dense(FFN // D * width, activation="relu"), Dense(width, activation="linear", bias=True))
            ),
        ]
    return [
        Embedding(tokens, width),
        Position(),
        *blocks,
        LayerNorm(),
        Dense(classes, output=True, activation="softmax", loss="cross_entropy"),
    ]


def _branch_block(width: int, heads: int) -> tuple[LayerSpec, ...]:
    # a branch: W11's block, its attention of heads heads, and a LayerNorm ending it
    return (
        StateInput(),
        Residual((LayerNorm(), Attention(heads=heads, causal=True))),
        Residual(
            (LayerNorm(), Dense(FFN // D * width, activation="relu"), Dense(width, activation="linear", bias=True))
        ),
        LayerNorm(),
    )


def branched_window(tokens: int, classes: int, factors: int, width: int = D, heads: int = 1) -> list[LayerSpec]:
    """
    The window branches workplan's window (D1, D2): W11's first two blocks as the shared trunk,
    then Branches, a block of heads heads (1, the retired design's; the post-paper workplan's
    branches-32 has 2, D8) and a LayerNorm for each of factors factors and one for the answers,
    which run on into the output layer; every path 3 blocks deep, width its stream's.
    """
    window = transformer_window(tokens, classes, width)
    trunk = window[: 2 + 3 * (BLOCKS - 1)]  # the Embedding, the Position and the first BLOCKS - 1 blocks
    return [*trunk, Branches(tuple(_branch_block(width, heads) for _ in range(factors + 1))), window[-1]]


def recombined_window(tokens: int, classes: int, factors: int, width: int = D) -> list[LayerSpec]:
    """
    The post-paper workplan's recombining window (D7, D8 (a), the owner 2026-10-27): W11's first two
    blocks as the shared trunk (as branched_window's), then merged Branches, a block of HEADS heads
    and a LayerNorm for each of factors factors, each factor's pooling reading its own branch's end;
    the ends summed, then the recombining block (W11's last block and its LayerNorm), which the
    output layer reads. Width its stream's. The factors' paths are 3 blocks deep, as the base
    window's and branched_window's (the trunk and their branch); the answers' 4 (the trunk, every
    branch, the recombining block), one more than the base's: the state's depth is held, the
    readout's isn't.
    """
    window = transformer_window(tokens, classes, width)
    block = 3  # a block's specs: its StateInput and two residual blocks
    trunk = window[: 2 + block * (BLOCKS - 1)]  # the Embedding, the Position and the first BLOCKS - 1 blocks
    recombining = window[-2 - block :]  # the last block, its LayerNorm and the output layer
    branches = Branches(tuple(_branch_block(width, HEADS) for _ in range(factors)), merge=True)
    return [*trunk, branches, *recombining]


def answer_branch_window(tokens: int, classes: int, factors: int, width: int = D) -> list[LayerSpec]:
    """
    The one-code-one-state workplan's R1 (D2 (a)): W11's first two blocks as the shared trunk, then
    two branches: the state branch, W11's third block unchanged (its heads, its FFN, its LayerNorm),
    which every factor's pooling reads, and the answers' own single-head block and LayerNorm, which
    runs on into the output layer; every path 3 blocks deep, width its stream's.
    """
    window = transformer_window(tokens, classes, width)
    trunk = window[: 2 + 3 * (BLOCKS - 1)]  # the Embedding, the Position and the first BLOCKS - 1 blocks
    state = tuple(window[len(trunk) : -1])  # the last block and its LayerNorm, as W11's
    answers = (
        StateInput(),
        Residual((LayerNorm(), Attention(heads=1, causal=True))),
        Residual(
            (LayerNorm(), Dense(FFN // D * width, activation="relu"), Dense(width, activation="linear", bias=True))
        ),
        LayerNorm(),
    )
    return [*trunk, Branches((state, answers), reads=(0,) * factors), window[-1]]


def linear_window(tokens: int, classes: int, width: int = D) -> list[LayerSpec]:
    """
    E6's baseline (the claims workplan, D21): no blocks, the state added to the embedded tokens and
    one linear residual (a StateInput stands before a residual block), then the output layer; no
    nonlinearity but the LayerNorms, so the recurrence does the step's work.
    """
    return [
        Embedding(tokens, width),
        Position(),
        StateInput(),
        Residual((LayerNorm(), Dense(width, activation="linear", bias=True))),
        LayerNorm(),
        Dense(classes, output=True, activation="softmax", loss="cross_entropy"),
    ]


def ffn_window(tokens: int, classes: int) -> list[LayerSpec]:
    """W11's window without its attention (E6's attention control): the state before each FFN."""
    blocks: list[LayerSpec] = []
    for _ in range(BLOCKS):
        blocks += [
            StateInput(),
            Residual((LayerNorm(), Dense(FFN, activation="relu"), Dense(D, activation="linear", bias=True))),
        ]
    return [
        Embedding(tokens, D),
        Position(),
        *blocks,
        LayerNorm(),
        Dense(classes, output=True, activation="softmax", loss="cross_entropy"),
    ]
