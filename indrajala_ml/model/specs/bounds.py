from collections.abc import Sized


def validate_input_bounds(dimension: int, input_bounds: list[tuple[float, float]]) -> None:
    """
    One (lo, hi) pair per input dimension, each of positive width: the precondition of every network
    that takes input_bounds.
    """

    assert len(input_bounds) == dimension
    assert all(hi > lo for lo, hi in input_bounds), f"input_bounds must all have positive width; got {input_bounds}"


def half_widths(input_bounds: list[tuple[float, float]]) -> list[float]:
    """
    Each dimension's (hi - lo) / 2.0, for LinearClassifierNetwork and BackpropClassifierNetwork.
    """
    return [(hi - lo) / 2.0 for lo, hi in input_bounds]


def validate_layer_sizes(layer_sizes: list[int], label: str = "layer_sizes", noun: str = "hidden layer") -> None:
    """
    At least one hidden layer, each of at least one node. label/noun name the argument in the
    message (the conv networks pass dense_layer_sizes).
    """
    assert len(layer_sizes) >= 1, f"{label} must specify at least one {noun}"
    assert all(size >= 1 for size in layer_sizes), f"every {noun} must have at least 1 node; got {layer_sizes}"


def validate_class_count(class_count: int) -> None:
    """
    A multiclass network needs at least 2 classes.
    """
    assert class_count >= 2, f"class_count must be at least 2; got {class_count}"


def validate_batch(batch: Sized) -> None:
    """
    A batch must hold at least one example.
    """
    assert len(batch) >= 1, "batch must not be empty"
