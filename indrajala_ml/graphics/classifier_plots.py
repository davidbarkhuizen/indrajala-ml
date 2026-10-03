# pyright: reportUnknownMemberType=false
# (matplotlib 3.8's annotations leave **kwargs untyped, so every Axes/Figure method reads as
# partially unknown; this module's own values are still checked where they're declared)
from collections.abc import Callable, Sequence

from matplotlib import lines
from matplotlib.axes import Axes

from indrajala_ml.geometry import reference_positive_region_polygon
from indrajala_ml.model.networks.python.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.model.protocols.classifier_protocols import StateClassifier


def plot_linear_classifier_network(
    axes: Axes,
    classifier: LinearClassifierNetwork,
    plotting_resolution: int = 100,
    color: str = "purple",
    x_bounds: tuple[float, float] | None = None,
):
    assert classifier.dimension == 2, (
        "plot_linear_classifier_network only supports 2D classifiers - each hidden node's "
        f"decision line is drawn in the x/y plane; got dimension={classifier.dimension}"
    )

    x_min, x_max = x_bounds if x_bounds else classifier.input_bounds[0]

    x_interval_size = x_max - x_min
    x_step_size = x_interval_size / float(plotting_resolution)
    x_ = [x_min + (i * x_step_size) for i in range(plotting_resolution)]

    for node in classifier.hidden_layer.nodes:
        a = node.input_node_weights[0]
        b = node.input_node_weights[1]
        c = node.threshold

        if b == 0.0 and a == 0.0:
            # a*x + b*y + c = 0 doesn't depend on position at all (the node is either always
            # active or always inactive, everywhere) - there's no line to draw for it
            continue
        elif b == 0.0:
            # a*x + c = 0 is vertical (no y term to solve for) - draw x = -c/a spanning the
            # axes' current y-range instead of dividing by the zero b below
            x_line = -c / a
            y_min, y_max = axes.get_ylim()
            line_graph = lines.Line2D([x_line, x_line], [y_min, y_max], color=color)
        else:
            y_ = [-1.0 * (a * x + c) / b for x in x_]
            line_graph = lines.Line2D(x_, y_, color=color)

        axes.add_line(line_graph)


def reference_region_bounds(
    classifier: LinearClassifierNetwork,
    fallback_bounds: list[tuple[float, float]],
    margin_fraction: float = 0.1,
) -> list[tuple[float, float]]:
    """
    fallback_bounds, expanded to contain the classifier's positive region when it is bounded
    (geometry.reference_positive_region_polygon); unchanged otherwise.
    """

    polygon = reference_positive_region_polygon(classifier)
    if not polygon:
        return fallback_bounds

    region_x = [x for x, _ in polygon]
    region_y = [y for _, y in polygon]
    region_x_min, region_x_max = min(region_x), max(region_x)
    region_y_min, region_y_max = min(region_y), max(region_y)

    margin_x = margin_fraction * max(region_x_max - region_x_min, 1.0e-9)
    margin_y = margin_fraction * max(region_y_max - region_y_min, 1.0e-9)

    (fallback_x_min, fallback_x_max), (fallback_y_min, fallback_y_max) = fallback_bounds

    return [
        (min(fallback_x_min, region_x_min - margin_x), max(fallback_x_max, region_x_max + margin_x)),
        (min(fallback_y_min, region_y_min - margin_y), max(fallback_y_max, region_y_max + margin_y)),
    ]


def plot_training_data(axes: Axes, training_data: Sequence[tuple[tuple[float, ...], float]]) -> None:

    markers = [".", "x"]
    colors = ["blue", "yellow"]

    categories: list[tuple[float, list[tuple[float, ...]]]] = []

    # sorted, so 0.0 gets the first marker and color and 1.0 the second
    for category_value in sorted({output_value for (_, output_value) in training_data}):
        categories.append(
            (category_value, [xy for (xy, output_value) in training_data if output_value == category_value])
        )

    for i in range(len(categories)):
        (category_value, values) = categories[i]
        marker = markers[i]
        color = colors[i]
        x, y = zip(*[(x_[0], x_[1]) for x_ in values])
        axes.plot(x, y, marker, color=color)


def plot_classifier_probability_heatmap(
    axes: Axes,
    classifier: StateClassifier[float],
    bounds: list[tuple[float, float]],
    resolution: int = 150,
) -> None:
    """
    The classifier's positive-class probability as a heatmap over bounds, for classifiers whose
    region isn't a union of half-planes (e.g. BackpropClassifierNetwork). Uses predict_probability
    when there is one, else classify_state's 0/1.
    """

    predict: Callable[[tuple[float, ...]], float] = getattr(
        classifier, "predict_probability", classifier.classify_state
    )

    (x_min, x_max), (y_min, y_max) = bounds
    x_step = (x_max - x_min) / float(resolution)
    y_step = (y_max - y_min) / float(resolution)

    # imshow expects rows top-to-bottom, so build the grid from y_max down to y_min
    grid = [
        [predict((x_min + col * x_step, y_max - row * y_step)) for col in range(resolution)]
        for row in range(resolution)
    ]

    axes.imshow(grid, extent=(x_min, x_max, y_min, y_max), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
