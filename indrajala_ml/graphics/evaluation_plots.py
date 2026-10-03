# pyright: reportUnknownMemberType=false
# (matplotlib 3.8's annotations leave **kwargs untyped, so every Axes/Figure method reads as
# partially unknown; this module's own values are still checked where they're declared)
import math
from collections.abc import Callable

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from indrajala_ml.graphics.chart import new_axes, new_figure


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


def new_confusion_matrix_figure(title: str, matrix: list[list[int]], class_labels: list[str] | None = None) -> Figure:
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
