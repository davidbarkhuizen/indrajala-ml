import math

import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.evaluate import agreement_label, class_balanced_disagreement_rate, compare_on_random_point, smoothed_series
from indrajala_ml.geometry import square_bounds
from indrajala_ml.graphics.chart import new_convergence_chart_pair, plot_labeled_series
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.train import random_alternating_training_data, train_linear_classifier_network


class StripesTarget:
    """
    Vertical stripes, alternating class every cell_size units of x (y is irrelevant) - unlike
    targets.py's XORTarget (2 regions), this target's number of regions
    scales with how wide input_bounds is relative to cell_size, so it's a convenient dial for
    "how hard is this problem" independent of any classifier's architecture.
    """

    def __init__(self, bounds: list[tuple[float, float]], cell_size: float = 2.0) -> None:
        self.input_bounds = bounds
        self.cell_size = cell_size

    def classify_state(self, state: tuple[float, float]) -> float:
        x, _ = state
        return 1.0 if math.floor(x / self.cell_size) % 2 == 0 else 0.0


def main() -> None:

    # [8] and [4, 4] share the same total node count, as do [4] and [8, 8]'s first layer vs
    # second - lets the printed results speak to depth vs width at a matched budget, rather
    # than just "more capacity wins"
    architectures: list[list[int]] = [[4], [8], [4, 4], [8, 8]]
    colors = ["yellow", "cyan", "magenta", "orange"]

    dimension = 2
    bounds = square_bounds(8.0, dimension)
    target = StripesTarget(bounds, cell_size=4.0)  # 4 alternating stripes across input_bounds

    learning_rate = 1.0
    training_set_size = 600
    epoch_count = 25

    print(
        "StripesTarget: 4 alternating vertical bands - a genuinely harder target than "
        "demo_xor_backprop_convergence.py's 2-region XOR, used here to actually compare architectures "
        "(including, for the first time in this demo set, a genuine 2-hidden-layer network) "
        "rather than just show any one of them succeeding. Unseeded, like every other sweep "
        "demo in this codebase - which architecture comes out ahead varies noticeably between "
        "runs (measured: sometimes the single wider layers win, sometimes the two-hidden-layer "
        "ones do), since this plain fixed-learning-rate gradient descent is sensitive to where "
        "random initialization happens to land."
    )
    print()

    training_data = random_alternating_training_data(training_set_size, target)

    results: list[tuple[str, str, list[int], list[float]]] = []

    for layer_sizes, color in zip(architectures, colors):
        label = "x".join(str(size) for size in layer_sizes)
        student = BackpropClassifierNetwork.randomized(layer_sizes, dimension, bounds)

        convergence_series = train_linear_classifier_network(
            student,
            training_data,
            learning_rate=learning_rate,
            epochs=epoch_count,
            reference_classifier=target,
        )

        n = [x[0] for x in convergence_series]
        disagreement = [x[1] for x in convergence_series]
        results.append((label, color, n, disagreement))

        diagnostic = convergence_series.diagnostic
        status = diagnostic.status_label
        final_disagreement = class_balanced_disagreement_rate(target, student, per_class_sample_count=300)

        new_state, reference_category, student_category = compare_on_random_point(target, student)
        agreement = agreement_label(reference_category, student_category)
        print(
            f"layer_sizes={label}: training accuracy {diagnostic.best_training_accuracy:.3f} "
            f"({status}), disagreement vs target {final_disagreement:.3f}, prediction on new "
            f"point: target={reference_category}, student={student_category} ({agreement})"
        )

    x_max = float(training_set_size * epoch_count)

    smoothed_results = [
        (label, color, n, smoothed_series(disagreement)) for label, color, n, disagreement in results
    ]

    print()
    print(
        "smoothed convergence chart: x-axis = training iteration (pooled across epochs), "
        "y-axis = each architecture's disagreement-rate series smoothed with a trailing "
        "moving average, same as demo_linear_classifier_cardinality_sweep.py - here comparing architectures "
        "instead of cardinalities"
    )

    smoothed_axes, smoothed_log_axes = new_convergence_chart_pair(
        "backprop convergence by architecture (smoothed)",
        "backprop convergence by architecture (smoothed, log scale)",
        x_max,
    )
    plot_labeled_series(smoothed_axes, smoothed_results)
    plot_labeled_series(smoothed_log_axes, smoothed_results)

    pyplot.show()


if __name__ == "__main__":
    main()
