# pyright: reportUnknownMemberType=false
# (matplotlib's untyped **kwargs, as in graphics/chart.py)
import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.evaluate import (
    agreement_label,
    class_balanced_disagreement_rate,
    compare_on_random_point,
    smoothed_series,
)
from indrajala_ml.geometry import square_bounds
from indrajala_ml.graphics.chart import new_convergence_chart_pair, plot_labeled_series
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.train import reachable_reference_and_training_data, train_linear_classifier_network


def main() -> None:

    cardinalities = [1, 2, 3, 4]
    colors = ["yellow", "cyan", "magenta", "orange"]

    dimension: int = 2
    l: float = 10.0
    bounds = square_bounds(l, dimension)

    learning_rate: float = 0.25
    training_set_size: int = 600
    epoch_count: int = 5

    results: list[tuple[str, str, list[int], list[float]]] = []

    for cardinality, color in zip(cardinalities, colors):
        reference, training_data = reachable_reference_and_training_data(
            cardinality, dimension, bounds, training_set_size
        )

        student = LinearClassifierNetwork.randomized(cardinality, dimension, bounds)

        convergence_series = train_linear_classifier_network(
            student,
            training_data,
            learning_rate=learning_rate,
            epochs=epoch_count,
            reference_classifier=reference,
        )

        n = [x[0] for x in convergence_series]
        disagreement = [x[1] for x in convergence_series]
        results.append((f"cardinality={cardinality}", color, n, disagreement))

        # measured on the final (pocketed) student, not convergence_series[-1][1], which is the
        # last training step's before the rollback to the best epoch (seed 19, cardinality=4:
        # 0.250 there, 0.017 for the final student)
        final_disagreement = class_balanced_disagreement_rate(reference, student, per_class_sample_count=300)

        _new_state, reference_category, student_category = compare_on_random_point(reference, student)
        agreement = agreement_label(reference_category, student_category)
        print(
            f"cardinality={cardinality}: disagreement {convergence_series[0][1]:.3f} -> "
            f"{final_disagreement:.3f}, prediction on new point: reference={reference_category}, "
            f"student={student_category} ({agreement})"
        )

    x_max = float(training_set_size * epoch_count)

    smoothed_results = [(label, color, n, smoothed_series(disagreement)) for label, color, n, disagreement in results]

    print(
        "smoothed convergence chart: x-axis = training iteration (pooled across epochs), "
        "y-axis = each cardinality's disagreement-rate series passed through a trailing "
        "moving average - makes the underlying trend easier to see through the sampling "
        "noise from class_balanced_disagreement_rate's small per-class sample count"
    )

    print(
        "smoothed log-scale convergence chart: the same smoothed curves as above, with a "
        "log-scaled y-axis - useful for comparing how fast each cardinality converges, "
        "since disagreement tends to drop roughly exponentially; a curve stops early once "
        "its disagreement reaches exactly zero, which has no position on a log axis"
    )

    smoothed_axes, smoothed_log_axes = new_convergence_chart_pair(
        "convergence by cardinality (smoothed)", "convergence by cardinality (smoothed, log scale)", x_max
    )
    plot_labeled_series(smoothed_axes, smoothed_results)
    plot_labeled_series(smoothed_log_axes, smoothed_results)

    pyplot.show()


if __name__ == "__main__":
    main()
