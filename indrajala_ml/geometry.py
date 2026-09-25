from indrajala_ml.model.classifier_protocols import TargetClassifier
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
    The classifier's positive region, the intersection of its hidden nodes' half-planes, as polygon
    vertices when bounded; an empty list when unbounded or empty.

    Only for an AND-combined classifier (required_active == cardinality): otherwise the region is a
    union of such intersections, and the polygon would be wrong, so it raises. Only for dimension
    == 2: the clipping reads each node's first two weights, so a higher-dimensional region (e.g. an
    infinite prism) could be reported bounded.

    huge defaults to a value derived from input_bounds, so the starting square's corners never
    coincide with a real vertex at any bounds scale; a fixed constant would misreport a large
    bounded region as unbounded.
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
    classifier: TargetClassifier[float], margin_fraction: float = 0.1
) -> list[tuple[float, float]] | None:
    """
    A tight axis-aligned box around the classifier's positive region, expanded by margin_fraction
    each side and clipped to input_bounds, or None when the region isn't computable
    (reference_positive_region_polygon), isn't bounded, or doesn't overlap input_bounds.

    The region is often under 1% of input_bounds at cardinality 4 (as low as 0.02%), so sampling
    positive points from this box (evaluate.sample_class_balanced_states) makes near-unreachable
    draws near-certain.

    classifier needs only input_bounds and classify_state; for anything but a
    LinearClassifierNetwork (whose dimension/required_active/cardinality it reads) this returns None.
    """

    if not isinstance(classifier, LinearClassifierNetwork):
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
