import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.geometry import square_bounds
from indrajala_ml.graphics.chart import new_axes, new_figure, plot_classifier_probability_heatmap, plot_training_data
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.train import random_alternating_training_data, train_linear_classifier_network


class CircleTarget:
    """
    A circular positive region. A LinearClassifierNetwork's region is always a polygon, so it can
    only facet a circle; a BackpropClassifierNetwork's sigmoid boundary can curve.
    """

    def __init__(self, bounds: list[tuple[float, float]], radius: float = 4.0) -> None:
        self.input_bounds = bounds
        self.radius = radius

    def classify_state(self, state: tuple[float, ...]) -> float:
        x, y = state
        return 1.0 if (x * x + y * y) < self.radius * self.radius else 0.0


def main() -> None:

    dimension = 2
    bounds = square_bounds(8.0, dimension)
    target = CircleTarget(bounds, radius=4.0)

    print(
        "Circular target: a disk of radius 4, centered on the origin - a genuinely curved "
        "boundary, not a polygon. A LinearClassifierNetwork's positive region is always an "
        "intersection of straight half-planes (see geometry.reference_positive_region_polygon), "
        "so it can only facet a circle with more and shorter straight edges, never truly curve; "
        "BackpropClassifierNetwork's sigmoid output can."
    )
    print()

    training_data = random_alternating_training_data(1000, target)

    student = BackpropClassifierNetwork.randomized([8], dimension, bounds)
    result = train_linear_classifier_network(student, training_data, learning_rate=1.0, epochs=80)

    diagnostic = result.diagnostic
    status = diagnostic.status_label
    print(
        f"training accuracy {diagnostic.best_training_accuracy:.3f} ({status} - best epoch "
        f"{diagnostic.best_epoch_index + 1}/{len(diagnostic.epoch_training_accuracies)})"
    )

    figure = new_figure("Backprop circular target: training data (true label) vs. learned probability heatmap")
    axes = new_axes(figure, bounds)
    plot_classifier_probability_heatmap(axes, student, bounds)
    plot_training_data(axes, training_data)
    pyplot.show()


if __name__ == "__main__":
    main()
