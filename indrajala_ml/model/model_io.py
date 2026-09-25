import json
from collections.abc import Sequence
from typing import Any, Protocol


class SupportsToList(Protocol):
    """A numpy or Rust array, as the array networks' snapshots hold them."""

    def tolist(self) -> Any: ...


def save_json(path: str, state: dict) -> None:
    """
    Writes state as JSON: for a save() whose envelope isn't save_model_json's (the conv networks,
    the array ensembles).
    """
    with open(path, "w") as f:
        json.dump(state, f)


def load_json(path: str) -> dict:
    """
    Reads a JSON file, with no envelope assumptions.
    """
    with open(path) as f:
        return json.load(f)


def save_model_json(
    path: str,
    *,
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    class_count: int,
    snapshot: object,
) -> None:
    """
    The envelope of the pure-Python multiclass network and ensemble: layer_sizes, dimension,
    input_bounds and class_count, which rebuild the network's shape, plus snapshot.
    """

    save_json(
        path,
        {
            "layer_sizes": layer_sizes,
            "dimension": dimension,
            "input_bounds": input_bounds,
            "class_count": class_count,
            "snapshot": snapshot,
        },
    )


def load_model_json(path: str) -> dict:
    """
    Reads save_model_json's envelope, with input_bounds turned back into tuples (JSON saves them as
    lists).
    """

    state = load_json(path)
    state["input_bounds"] = [tuple(bound) for bound in state["input_bounds"]]
    return state


def save_array_model_json(
    path: str,
    *,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    snapshot: Sequence[tuple[SupportsToList, SupportsToList]],
    extra: dict | None = None,
) -> None:
    """
    The envelope of every multiclass array network, numpy and Rust: layer_sizes, dimension,
    class_count and the snapshot, with extra (a sibling's hyperparameters, e.g. Adam's
    beta1/beta2/epsilon) merged in. The array networks have no input_bounds. snapshot is the
    network's (W, b) list, converted to lists here.
    """

    state = {
        "layer_sizes": layer_sizes,
        "dimension": dimension,
        "class_count": class_count,
        "snapshot": [(W.tolist(), b.tolist()) for W, b in snapshot],
    }
    if extra:
        state.update(extra)
    save_json(path, state)


def load_array_model_json(path: str) -> dict:
    """
    Reads save_array_model_json's envelope; the network's restore() converts the snapshot through
    its backend.
    """

    return load_json(path)


def save_single_output_array_model_json(
    path: str,
    *,
    layer_sizes: list[int],
    dimension: int,
    snapshot: Sequence[tuple[SupportsToList, SupportsToList]],
    extra: dict | None = None,
) -> None:
    """
    save_array_model_json without class_count, for the single-output array networks (ensemble
    sub-networks).
    """

    state = {
        "layer_sizes": layer_sizes,
        "dimension": dimension,
        "snapshot": [(W.tolist(), b.tolist()) for W, b in snapshot],
    }
    if extra:
        state.update(extra)
    save_json(path, state)


def load_single_output_array_model_json(path: str) -> dict:
    """
    Reads save_single_output_array_model_json's envelope.
    """

    return load_json(path)
