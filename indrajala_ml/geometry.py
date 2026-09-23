from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork


def square_bounds(l: float, dimension: int = 2) -> list[tuple[float, float]]:
    return [(-l, l)] * dimension


def _clip_polygon_by_halfplane(
    polygon: list[tuple[float, float]], a: float, b: float, c: float
) -> list[tuple[float, float]]:

    # Sutherland-Hodgman clipping against the half-plane a*x + b*y + c > 0 (the same
    # inequality AssociationNode.z() > 0 tests for a hidden node's weights/threshold)
    def signed_distance(point: tuple[float, float]) -> float:
        return a * point[0] + b * point[1] + c

    output: list[tuple[float, float]] = []

    for i in range(len(polygon)):
        current = polygon[i]
        previous = polygon[i - 1]

        current_distance = signed_distance(current)
        previous_distance = signed_distance(previous)

        current_inside = current_distance > 0.0
        previous_inside = previous_distance > 0.0

        if current_inside != previous_inside:
            t = previous_distance / (previous_distance - current_distance)
            output.append(
                (
                    previous[0] + t * (current[0] - previous[0]),
                    previous[1] + t * (current[1] - previous[1]),
                )
            )

        if current_inside:
            output.append(current)

    return output


def reference_positive_region_polygon(
    classifier: LinearClassifierNetwork, huge: float | None = None
) -> list[tuple[float, float]]:
    """
    The classifier's positive region is the intersection of its hidden nodes' half-planes,
    which may be bounded (closed, e.g. a triangle or other convex polygon) or unbounded
    (e.g. any single half-plane, or several whose intersection still extends to infinity).

    Returns the region's polygon vertices when it's bounded; an empty list when it's
    unbounded or empty.

    Only valid for an AND-combined classifier (required_active == cardinality) - the
    intersection-of-half-planes computed here is only actually the classifier's positive
    region under AND. For any other required_active (e.g. OR, or a general k-of-n gate),
    the true positive region is a union of such intersections, which this function does not
    compute; calling it on one would silently return a wrong polygon, so it's rejected
    outright instead.

    Also only valid for dimension == 2 - the polygon-clipping below only ever reads each
    hidden node's first two weights, so a classifier with more dimensions would otherwise be
    silently projected onto the first two and could report a completely wrong answer (e.g. a
    genuinely unbounded higher-dimensional region, like an infinite prism, reported as
    bounded because the dimensions extending it to infinity were never even looked at).

    huge defaults to a value derived from classifier.input_bounds (large enough that the
    initial clipping square's corners can never coincide with a real, bounded region's
    vertices) rather than a fixed constant - a fixed "large enough" absolute constant would
    itself be wrong at a large enough input_bounds scale, incorrectly reporting a genuinely
    bounded but large region as unbounded once its vertices approach that fixed constant.
    """

    assert classifier.dimension == 2, (
        "reference_positive_region_polygon only supports 2D classifiers - it only reads "
        f"each hidden node's first two weights; got dimension={classifier.dimension}"
    )

    assert classifier.required_active == classifier.cardinality, (
        "reference_positive_region_polygon only supports AND-combined classifiers "
        f"(required_active == cardinality); got required_active={classifier.required_active} "
        f"with cardinality={classifier.cardinality}"
    )

    if huge is None:
        # 1.0e5x the largest half-width - at the half-width of 10 every existing demo and
        # test uses, this equals 1.0e6
        huge = 1.0e5 * max(classifier.half_widths())

    polygon: list[tuple[float, float]] = [(-huge, -huge), (huge, -huge), (huge, huge), (-huge, huge)]

    for node in classifier.hidden_layer.nodes:
        a, b, c = node.input_node_weights[0], node.input_node_weights[1], node.threshold
        polygon = _clip_polygon_by_halfplane(polygon, a, b, c)
        if not polygon:
            return []

    if any(abs(x) >= huge * 0.99 or abs(y) >= huge * 0.99 for x, y in polygon):
        return []

    return polygon


def is_positive_region_bounded(classifier: LinearClassifierNetwork) -> bool:
    return bool(reference_positive_region_polygon(classifier))


def positive_region_bounding_box(
    classifier: LinearClassifierNetwork, margin_fraction: float = 0.1
) -> list[tuple[float, float]] | None:
    """
    A tight axis-aligned box around classifier's positive region, expanded by
    margin_fraction on each side and clipped to classifier.input_bounds - or None when the
    region isn't computable this way (dimension != 2, a non-AND combination - see
    reference_positive_region_polygon) or isn't bounded, or when clipping to input_bounds
    leaves nothing (the region doesn't actually overlap input_bounds).

    The positive region can be a tiny fraction of input_bounds - verified empirically: often
    under 1% of the bounding box's area at cardinality 4, as low as 0.02% - so sampling
    positive-class points uniformly from this tight box instead of the whole input_bounds
    (see evaluate.sample_class_balanced_states) turns what can be a near-unreachable rate of
    positive draws into a near-certain one.

    classifier only needs the same duck-typed input_bounds/classify_state interface
    train.py/evaluate.py's other functions require - dimension/required_active/cardinality
    (LinearClassifierNetwork-specific) are optional, and their absence is treated the same as
    not being a 2D AND-combined classifier: this function just returns None.
    """

    if not all(hasattr(classifier, attribute) for attribute in ("dimension", "required_active", "cardinality")):
        return None

    if classifier.dimension != 2 or classifier.required_active != classifier.cardinality:
        return None

    polygon = reference_positive_region_polygon(classifier)
    if not polygon:
        return None

    region_x = [x for x, _ in polygon]
    region_y = [y for _, y in polygon]
    margin_x = margin_fraction * max(max(region_x) - min(region_x), 1.0e-9)
    margin_y = margin_fraction * max(max(region_y) - min(region_y), 1.0e-9)

    (input_x_min, input_x_max), (input_y_min, input_y_max) = classifier.input_bounds
    box = [
        (max(input_x_min, min(region_x) - margin_x), min(input_x_max, max(region_x) + margin_x)),
        (max(input_y_min, min(region_y) - margin_y), min(input_y_max, max(region_y) + margin_y)),
    ]
    if any(lo >= hi for lo, hi in box):
        return None

    return box
