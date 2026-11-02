"""
Converting a trained model to grouped-query attention (the grouped-query attention workplan, D5;
Ainslie et al. 2023, "GQA", §2.1): to_grouped takes a format-2 document (network_to_json's, any
implementation) whose attention layers give each query head its own key/value head, and returns
one whose layers share g key/value heads, each group's key and value projections the mean of its
heads': per row offset within a head, a left fold over the group's heads in head order, from the
first, then divided by the group's size h / g. The queries' and the output's projections are kept.
Ainslie et al. then "uptrain" the converted model for a fraction of its original steps.

The converted document has a fresh optimizer state (t 0, every layer's state made on its first
step): the rule's moments for the old Wk and Wv have no counterpart in the grouped ones. Its
generator state is kept. A layer whose heads are g already is kept as it is, so g = heads returns
the document unchanged.

Backend-free, as format2.py: it works on the document's nested lists.
"""

from __future__ import annotations

import copy
from typing import Any, cast

from indrajala_ml.model.persistence.format2 import PYTHON, WINDOW_RECURRENT, is_format2
from indrajala_ml.model.persistence.format2_json import layer_from_json
from indrajala_ml.model.specs.layer_specs import Attention, expand_specs


def _mean_blocks(rows: list[Any], heads: int, key_value_heads: int) -> list[Any]:
    # rows in heads blocks of d_k (each row a list of weights, or a bias's float), as key_value_heads
    # blocks: block j's row r the left fold of heads j * (h / g) .. (j + 1) * (h / g) - 1's rows r, in
    # head order from the first, divided by h / g
    d_k = len(rows) // heads
    size = heads // key_value_heads
    grouped: list[Any] = []
    for j in range(key_value_heads):
        for r in range(d_k):
            members = [rows[(j * size + m) * d_k + r] for m in range(size)]
            if isinstance(members[0], list):
                vectors = cast("list[list[float]]", members)
                total = list(vectors[0])
                for member in vectors[1:]:
                    total = [a + b for a, b in zip(total, member, strict=True)]
                grouped.append([x / size for x in total])
            else:
                scalars = cast("list[float]", members)
                value = scalars[0]
                for member in scalars[1:]:
                    value += member
                grouped.append(value / size)
    return grouped


def _group_layer(weights: list[Any], heads: int, key_value_heads: int, python: bool) -> list[Any]:
    # one attention layer's weights entry with Wk, bk, Wv and bv grouped
    if not python:
        Wq, bq, Wk, bk, Wv, bv, Wo, bo = weights
        return [
            Wq,
            bq,
            _mean_blocks(Wk, heads, key_value_heads),
            _mean_blocks(bk, heads, key_value_heads),
            _mean_blocks(Wv, heads, key_value_heads),
            _mean_blocks(bv, heads, key_value_heads),
            Wo,
            bo,
        ]
    # pure Python: [weights, bias] per row, Wq's h * d_k rows, then Wk's, Wv's, then Wo's d rows
    features = len(weights[0][0])
    width = (len(weights) - features) // 3
    queries, keys, values, outputs = (
        weights[:width],
        weights[width : 2 * width],
        weights[2 * width : 3 * width],
        weights[3 * width :],
    )

    def grouped(rows: list[Any]) -> list[Any]:
        new_weights = _mean_blocks([row[0] for row in rows], heads, key_value_heads)
        new_biases = _mean_blocks([row[1] for row in rows], heads, key_value_heads)
        return [[w, b] for w, b in zip(new_weights, new_biases, strict=True)]

    return [*queries, *grouped(keys), *grouped(values), *outputs]


def _group_specs(entry: Any, key_value_heads: int) -> Any:
    # the layers' JSON with each attention entry's key_value_heads set (left out where it's the heads)
    if isinstance(entry, list):
        return [_group_specs(item, key_value_heads) for item in cast("list[Any]", entry)]
    if not isinstance(entry, dict):
        return entry
    new: dict[str, Any] = {
        key: _group_specs(value, key_value_heads) for key, value in cast("dict[str, Any]", entry).items()
    }
    if new.get("kind") == "attention" and new.get("heads", 1) != key_value_heads:
        new["key_value_heads"] = key_value_heads
    return new


def to_grouped(document: dict[str, Any], key_value_heads: int) -> dict[str, Any]:
    """
    document (a format-2 network's) with every attention layer's heads sharing key_value_heads
    key/value heads, their projections mean-pooled per group (D5); a fresh optimizer state.
    Raises ValueError for a document that isn't one network's, a window-recurrent one (its window's
    attention isn't grouped, D10), an attention already grouped, or key_value_heads not dividing
    an attention's heads.
    """
    if not is_format2(document) or document.get("shape") not in ("multiclass", "single_output", "sequence"):
        shape = document.get("shape")
        reason = "a window's attention isn't grouped (D10)" if shape == WINDOW_RECURRENT else "not one network's"
        raise ValueError(
            f"to_grouped converts one sequential network's format-2 document; got shape {shape!r}: {reason}"
        )
    specs = expand_specs([layer_from_json(entry) for entry in document["layers"]])
    attention = [(index, spec) for index, spec in enumerate(specs) if isinstance(spec, Attention)]
    for _, spec in attention:
        if spec.key_value_heads is not None:
            raise ValueError(
                f"to_grouped converts attention whose query heads each have their own key/value head; got {spec!r}"
            )
        if key_value_heads < 1 or spec.heads % key_value_heads != 0:
            raise ValueError(f"key_value_heads {key_value_heads} doesn't divide the heads of {spec!r}")
    changed = [(index, spec) for index, spec in attention if spec.heads != key_value_heads]
    if not changed:
        return document
    converted = copy.deepcopy(document)
    python = document["implementation"] == PYTHON
    for index, spec in changed:
        converted["weights"][index] = _group_layer(converted["weights"][index], spec.heads, key_value_heads, python)
    converted["layers"] = _group_specs(converted["layers"], key_value_heads)
    converted["optimizer_state"] = {"t": 0, "layers": [None] * len(document["optimizer_state"]["layers"])}
    return converted
