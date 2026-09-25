import math
from typing import Callable

from matplotlib import lines, pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend

from indrajala_ml.geometry import reference_positive_region_polygon
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork


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


def plot_training_data(axes: Axes, training_data: list[tuple[tuple[float, float], float]]):

    markers = [".", "x"]
    colors = ["blue", "yellow"]

    categories: list[tuple[int, list[tuple[float, float]]]] = []

    # sorted, so 0.0 gets the first marker and color and 1.0 the second
    for category_value in sorted(set(output_value for (_, output_value) in training_data)):
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
    classifier,
    bounds: list[tuple[float, float]],
    resolution: int = 150,
) -> None:
    """
    The classifier's positive-class probability as a heatmap over bounds, for classifiers whose
    region isn't a union of half-planes (e.g. BackpropClassifierNetwork). Uses predict_probability
    when there is one, else classify_state's 0/1.
    """

    predict = getattr(classifier, "predict_probability", classifier.classify_state)

    (x_min, x_max), (y_min, y_max) = bounds
    x_step = (x_max - x_min) / float(resolution)
    y_step = (y_max - y_min) / float(resolution)

    # imshow expects rows top-to-bottom, so build the grid from y_max down to y_min
    grid = [
        [predict((x_min + col * x_step, y_max - row * y_step)) for col in range(resolution)]
        for row in range(resolution)
    ]

    axes.imshow(grid, extent=(x_min, x_max, y_min, y_max), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")


def plot_confusion_matrix(axes: Axes, matrix: list[list[int]], class_labels: list[str] | None = None) -> None:
    """
    A confusion matrix (matrix[true][predicted] = count) as a heatmap with each cell's count.
    """

    class_count = len(matrix)
    class_labels = class_labels if class_labels is not None else [str(i) for i in range(class_count)]

    axes.imshow(matrix, cmap="viridis")
    axes.set_xticks(range(class_count))
    axes.set_yticks(range(class_count))
    axes.set_xticklabels(class_labels)
    axes.set_yticklabels(class_labels)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")

    row_max = [max(row) if row else 0 for row in matrix]
    for true_label in range(class_count):
        for predicted_label in range(class_count):
            count = matrix[true_label][predicted_label]
            # dark text on the heatmap's bright cells, light text on its dark ones
            text_color = "black" if row_max[true_label] and count > row_max[true_label] / 2 else "white"
            axes.text(predicted_label, true_label, str(count), ha="center", va="center", color=text_color)


def new_confusion_matrix_figure(
    title: str, matrix: list[list[int]], class_labels: list[str] | None = None
) -> Figure:
    """
    A new figure and axes with plot_confusion_matrix drawn on it, as every recognition demo shows.
    """

    figure = new_figure(title)
    axes = new_axes(figure, scaled=False)
    plot_confusion_matrix(axes, matrix, class_labels)
    return figure


def sample_predictions_figure(
    title: str,
    test_data: list[tuple[tuple[float, ...], int]],
    classify_fn: Callable[[tuple[float, ...]], int],
    count: int = 16,
    image_shape: tuple[int, int] = (8, 8),
) -> Figure:
    """
    Classifies the first count test examples with classify_fn and plots them with
    plot_sample_predictions.
    """

    sample_count = min(count, len(test_data))
    samples = [(state, classify_fn(state), true_label) for state, true_label in test_data[:sample_count]]
    figure = new_figure(title)
    plot_sample_predictions(figure, samples, image_shape=image_shape)
    return figure


def plot_sample_predictions(
    figure: Figure,
    samples: list[tuple[tuple[float, ...], int, int]],
    image_shape: tuple[int, int] = (8, 8),
) -> None:
    """
    A grid of subplots, one image per (pixels, predicted_label, true_label) sample, titled white
    when correct and red when not. Any image_shape.
    """

    columns = math.ceil(math.sqrt(len(samples)))
    rows = math.ceil(len(samples) / columns)
    height, width = image_shape

    for index, (pixels, predicted_label, true_label) in enumerate(samples):
        axes = figure.add_subplot(rows, columns, index + 1)
        image = [pixels[row * width : (row + 1) * width] for row in range(height)]
        axes.imshow(image, cmap="gray")
        axes.set_xticks([])
        axes.set_yticks([])
        correct = predicted_label == true_label
        axes.set_title(f"pred={predicted_label} true={true_label}", color="white" if correct else "red", fontsize=8)


def style_dark_legend(legend: Legend) -> None:
    """
    Styles a legend for the dark chart theme: a legend's frame and text stay light whatever the
    axes' facecolor.
    """

    legend.get_frame().set_facecolor("black")
    for text in legend.get_texts():
        text.set_color("white")


def plot_labeled_series(axes: Axes, results: list[tuple[str, str, list[float], list[float]]]) -> None:
    """
    Plots each (label, color, x, y) series on axes, with a dark legend: the multi-series convergence
    chart of the sweep demos.
    """

    for label, color, x, y in results:
        axes.plot(x, y, color=color, label=label)

    style_dark_legend(axes.legend())


def disagreement_axis_bounds(x_max: float, log: bool = False) -> list[tuple[float, float]]:
    # a disagreement rate is always in [0, 1]; on a log-scaled axis 0.0 has no position, so
    # floor it just above zero instead
    return [(0.0, x_max), (1.0e-3, 1.0) if log else (0.0, 1.0)]


def new_convergence_chart_pair(linear_title: str, log_title: str, x_max: float) -> tuple[Axes, Axes]:
    """
    Linear and log-scaled disagreement axes, each on a new figure, for a convergence chart;
    callers plot on both.
    """

    linear_axes = new_axes(new_figure(linear_title), disagreement_axis_bounds(x_max), scaled=False)

    log_axes = new_axes(new_figure(log_title), disagreement_axis_bounds(x_max, log=True), scaled=False)
    log_axes.set_yscale("log")

    return linear_axes, log_axes


def new_figure(label: str) -> Figure:
    figure = pyplot.figure(label)
    figure.patch.set_facecolor("xkcd:black")
    return figure


def new_axes(figure: Figure, bounds: list[tuple[float, float]] | None = None, scaled=True) -> Axes:

    axes = figure.add_subplot(111)
    axes.set_facecolor("xkcd:black")

    axes.grid(True, which="both")

    if bounds:
        axes.set_xlim(bounds[0])
        axes.set_ylim(bounds[1])

    if scaled:
        axes.set_aspect("equal", adjustable="box")

    axes.spines["bottom"].set_color("white")
    axes.spines["top"].set_color("white")
    axes.spines["left"].set_color("white")
    axes.spines["right"].set_color("white")

    axes.xaxis.label.set_color("white")
    axes.yaxis.label.set_color("white")
    axes.tick_params(axis="x", colors="white")
    axes.tick_params(axis="y", colors="white")

    return axes
