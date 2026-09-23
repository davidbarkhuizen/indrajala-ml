def validate_input_bounds(dimension: int, input_bounds: list[tuple[float, float]]) -> None:
    """
    Shared precondition behind every model class that takes an explicit input_bounds: one
    (lo, hi) pair per input dimension, each with strictly positive width - both
    LinearClassifierNetwork and BackpropNetworkBase (and so every backprop network class built
    on it) require exactly this.
    """

    assert len(input_bounds) == dimension
    assert all(hi > lo for lo, hi in input_bounds), f"input_bounds must all have positive width; got {input_bounds}"


def half_widths(input_bounds: list[tuple[float, float]]) -> list[float]:
    """
    Shared by both LinearClassifierNetwork.half_widths and
    BackpropClassifierNetwork.half_widths (no base class between the two to hang this on) -
    each dimension's (hi - lo) / 2.0.
    """
    return [(hi - lo) / 2.0 for lo, hi in input_bounds]


def validate_layer_sizes(layer_sizes: list[int], label: str = "layer_sizes", noun: str = "hidden layer") -> None:
    """
    Shared precondition behind every network's hidden-layer-shape constructor argument: at
    least one hidden layer, and every one of them at least one node. BackpropNetworkBase and
    every array-backed sibling (VectorizedMultiClassBackpropClassifierNetwork,
    RustArrayMultiClassBackpropClassifierNetwork, and their Adam counterparts - standalone
    classes, not BackpropNetworkBase subclasses, so this check would otherwise be copy-pasted
    once per class) all require exactly this. label/noun let
    ConvMultiClassBackpropClassifierNetwork reuse this for its own differently-named
    dense_layer_sizes argument without changing the message every other caller already shows.
    """
    assert len(layer_sizes) >= 1, f"{label} must specify at least one {noun}"
    assert all(size >= 1 for size in layer_sizes), f"every {noun} must have at least 1 node; got {layer_sizes}"


def validate_class_count(class_count: int) -> None:
    """
    Shared precondition behind every multi-class network's class_count constructor argument -
    at least 2 classes, the same reasoning square_bounds/half_widths above share across
    LinearClassifierNetwork/BackpropClassifierNetwork: a multi-class classifier over fewer than
    2 classes isn't a meaningful thing to build.
    """
    assert class_count >= 2, f"class_count must be at least 2; got {class_count}"


def validate_batch(batch) -> None:
    """
    Shared precondition behind every learn_batch/_learn_batch implementation - a batch must
    contain at least one example, the same check BackpropNetworkBase._learn_batch and every
    array-backed sibling's own learn_batch require independently.
    """
    assert len(batch) >= 1, "batch must not be empty"
