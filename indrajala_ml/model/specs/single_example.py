"""
Batch norm's refusals of what would normalize a batch of one example to 0 (the batch-norm workplan,
D4): a one-example training step, and a ghost group of one example (D6). Every implementation's
batch-norm layers and networks raise these, so the messages match.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from indrajala_ml.model.specs.layer_specs import BatchNorm, LayerSpec, expand_specs, spec_paths


def refuse_single_example(layer: object) -> NoReturn:
    """A batch-norm layer's, and its linear layer's, refusal of a one-example training step (the
    batch-norm workplan, D4), in every implementation."""
    raise ValueError(
        f"a {type(layer).__name__} trains on batches only: batch norm normalizes a batch of one to 0 "
        "(the batch-norm workplan, D4)"
    )


def refuse_single_example_network(specs: Sequence[LayerSpec], batch_norm_index: int) -> NoReturn:
    """A network's refusal of a one-example training step, naming its first batch-norm layer (D4),
    batch_norm_index an index into expand_specs(specs)."""
    spec = expand_specs(specs)[batch_norm_index]
    raise ValueError(
        f"{spec_paths(specs)[batch_norm_index]}, {spec!r}, can't train on one example: it "
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
    a last ghost group of one example (D6), naming the layer, a block's body's included."""
    for path, spec in zip(spec_paths(specs), expand_specs(specs), strict=True):
        group_size = spec.group_size if isinstance(spec, BatchNorm) else None
        if group_size is not None and batch_size > group_size and batch_size % group_size == 1:
            raise ValueError(f"{path}, {spec!r}: {_single_example_group(batch_size, group_size)}")


def batch_norm_index(specs: Sequence[LayerSpec]) -> int | None:
    """The index in expand_specs(specs) of the first BatchNorm, a block's body's included, if any:
    such a network refuses a one-example training step (D4)."""
    return next((i for i, spec in enumerate(expand_specs(specs)) if isinstance(spec, BatchNorm)), None)
