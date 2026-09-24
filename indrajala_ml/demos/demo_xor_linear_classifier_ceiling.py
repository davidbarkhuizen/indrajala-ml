import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.evaluate import class_balanced_disagreement_rate
from indrajala_ml.geometry import square_bounds
from indrajala_ml.graphics.chart import new_axes, new_figure, plot_linear_classifier_network, plot_training_data
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.targets import XORTarget
from indrajala_ml.train import random_alternating_training_data, train_linear_classifier_network


def main() -> None:

    dimension = 2
    bounds = square_bounds(10.0, dimension)
    target = XORTarget(bounds)

    training_data = random_alternating_training_data(1000, target)

    print(
        "XOR-style target: category = (x > 0) != (y > 0) - two diagonally opposite "
        "quadrants, a disconnected, non-convex region. Every other demo's target is a "
        "LinearClassifierNetwork of the same architecture the student trains with, so it's "
        "representable by construction; this one deliberately isn't."
    )
    print()

    # cardinality 1-4, swept across AND, OR, and (where distinct from both) majority gates -
    # the point isn't which gate is "best", it's that none of them get close to converging
    configs: list[tuple[int, int, str]] = [
        (1, 1, "cardinality=1"),
        (2, 2, "cardinality=2, AND"),
        (2, 1, "cardinality=2, OR"),
        (3, 3, "cardinality=3, AND"),
        (3, 1, "cardinality=3, OR"),
        (3, 2, "cardinality=3, majority (2-of-3)"),
        (4, 4, "cardinality=4, AND"),
        (4, 1, "cardinality=4, OR"),
        (4, 2, "cardinality=4, majority (2-of-4)"),
    ]

    best_student: LinearClassifierNetwork | None = None
    best_disagreement = float("inf")
    best_label = ""
    plateaued_count = 0
    converged_count = 0

    for cardinality, required_active, label in configs:
        student = LinearClassifierNetwork.randomized(cardinality, dimension, bounds, required_active)
        result = train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=10)
        disagreement = class_balanced_disagreement_rate(target, student, per_class_sample_count=300)

        diagnostic = result.diagnostic
        status = diagnostic.status_label
        plateaued_count += diagnostic.plateaued
        converged_count += diagnostic.converged
        print(
            f"{label}: final disagreement = {disagreement:.3f}, training accuracy "
            f"{diagnostic.best_training_accuracy:.3f} ({status} - best epoch "
            f"{diagnostic.best_epoch_index + 1}/{len(diagnostic.epoch_training_accuracies)})"
        )

        if disagreement < best_disagreement:
            best_disagreement = disagreement
            best_student = student
            best_label = label

    print()
    print(
        f"none come close to converging (best seen: {best_disagreement:.3f}, {best_label}): "
        f"{converged_count}/{len(configs)} configurations above reported 'converged', "
        f"{plateaued_count}/{len(configs)} 'plateaued' (more epochs already stopped helping "
        f"well before the last one), and the rest were still (slowly) improving as of the "
        "last epoch. Either way, more training isn't the fix: the output layer's weights "
        "are always 1.0 per hidden node, so the output can only be a monotonically "
        "non-decreasing function of how many hidden nodes are active. XOR needs the "
        "opposite for some units (one hidden node firing should sometimes make the output "
        "LESS likely to fire), which no choice of required_active, at any cardinality, can "
        "express."
    )

    figure = new_figure(f"XOR-style target: training data (true label) vs. best student's hyperplanes ({best_label})")
    axes = new_axes(figure, bounds)
    plot_training_data(axes, training_data)
    plot_linear_classifier_network(axes, best_student, color="purple", x_bounds=bounds[0])
    pyplot.show()


if __name__ == "__main__":
    main()
