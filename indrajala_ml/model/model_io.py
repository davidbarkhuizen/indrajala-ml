import json


def save_json(path: str, state: dict) -> None:
    """
    Bare open/json.dump wrapping, shared by every save() in this codebase whose envelope shape
    doesn't fit save_model_json's fixed layer_sizes/class_count fields (e.g.
    ConvMultiClassBackpropClassifierNetwork.save, which has conv hyperparameters instead).
    """
    with open(path, "w") as f:
        json.dump(state, f)


def load_json(path: str) -> dict:
    """The load-side counterpart to save_json - bare open/json.load, no envelope assumptions."""
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
    The shared JSON envelope behind MultiClassBackpropClassifierNetwork.save and
    EnsembleBackpropClassifierNetwork.save - both need exactly enough to reconstruct a
    network's shape (layer_sizes, dimension, input_bounds, class_count) plus its trained
    weights (snapshot); only what goes into snapshot and how many networks get built from this
    envelope differs between the two.
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
    The load-side counterpart to save_model_json: reads the envelope back, with input_bounds
    already restored to tuples (JSON only has arrays, so a saved (lo, hi) tuple round-trips as a
    2-element list otherwise) - everything else in the envelope is returned as-is for the caller
    to reconstruct its own network shape(s) from.
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
    snapshot: object,
    extra: dict | None = None,
) -> None:
    """
    The array-backed counterpart to save_model_json above - shared by every
    ArrayLayer/RustArrayLayer-backed sibling network's own save()
    (VectorizedMultiClassBackpropClassifierNetwork, RustArrayMultiClassBackpropClassifierNetwork,
    and their Adam counterparts): none of them has an input_bounds/StateLayer notion to save
    (see VectorizedMultiClassBackpropClassifierNetwork.save's own docstring for why this can't
    just be save_model_json), but every one needs the same layer_sizes/dimension/class_count/
    snapshot shape, plus room for a sibling-specific extra dict (e.g. Adam's own
    beta1/beta2/epsilon) merged in on top - snapshot is taken as the raw (W, b) array-pair list
    self.snapshot() already returns and converted to JSON-serializable lists here, once, rather
    than at each of the 4 call sites.
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
    The load-side counterpart to save_array_model_json - reads the envelope back as-is; the
    caller reconstructs snapshot arrays with its own backend-specific array constructor
    (numpy's np.array or indrajala_math_rust.Array), since this module has no array-backend
    dependency of its own.
    """

    return load_json(path)


def save_single_output_array_model_json(
    path: str,
    *,
    layer_sizes: list[int],
    dimension: int,
    snapshot: object,
    extra: dict | None = None,
) -> None:
    """
    The single-output counterpart to save_array_model_json above - shared by
    ArrayBackpropClassifierNetwork/RustArrayBackpropClassifierNetwork's own save(): neither has a
    class_count notion at all (each is one independent binary sub-network, not a multiclass
    output layer), on top of save_array_model_json's own already-missing input_bounds/StateLayer
    notion - so this drops that field rather than passing a meaningless class_count=1 through
    save_array_model_json.
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
    """The load-side counterpart to save_single_output_array_model_json - reads the envelope back as-is."""

    return load_json(path)
