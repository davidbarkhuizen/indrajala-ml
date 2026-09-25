import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.evaluate import agreement_label, compare_on_random_point, smoothed_series
from indrajala_ml.geometry import is_positive_region_bounded, square_bounds
from indrajala_ml.graphics.chart import (
    new_axes,
    new_convergence_chart_pair,
    new_figure,
    plot_linear_classifier_network,
    plot_training_data,
    reference_region_bounds,
)
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.train import reachable_reference_and_training_data, train_linear_classifier_network


def main() -> None:

    classifier_cardinality: int = 4
    dimension: int = 2
    l: float = 10.0

    learning_rate: float = 0.25
    training_set_size: int = 1000
    epoch_count: int = 1

    input_bounds = square_bounds(l, dimension)

    # a random reference classifier and its training data, redrawn when a class is unreachable.
    # A 2D region needs 3 half-planes to be bounded, so from cardinality 3 only a reference with
    # a bounded region is accepted
    #
    require_bounded_region = classifier_cardinality >= 3

    # random half-planes enclose a bounded region only ~10% of the time at cardinality=4, but
    # the check is pure geometry, so a large attempt budget stays fast
    reference_classifier, training_data = reachable_reference_and_training_data(
        classifier_cardinality,
        dimension,
        input_bounds,
        training_set_size,
        regeneration_attempts=200 if require_bounded_region else 20,
        is_valid=is_positive_region_bounded if require_bounded_region else None,
    )

    # generate a new random classifier network for training
    #
    student_classifier = LinearClassifierNetwork.randomized(classifier_cardinality, dimension, input_bounds)

    convergence_series: list[tuple[int, float]] = train_linear_classifier_network(
        student_classifier,
        training_data,
        learning_rate=learning_rate,
        epochs=epoch_count,
        reference_classifier=reference_classifier,
    )

    # use the trained student to classify a fresh point, never seen during training
    #
    new_state, reference_category, student_category = compare_on_random_point(reference_classifier, student_classifier)
    agreement = agreement_label(reference_category, student_category)
    print(
        f"prediction on new point {new_state}: reference={reference_category}, student={student_category} ({agreement})"
    )

    n: list[int] = [x[0] for x in convergence_series]
    disagreement: list[float] = smoothed_series([x[1] for x in convergence_series])

    print(
        "convergence chart (linear scale): x-axis = training iteration, y-axis = "
        "disagreement rate (fraction of sampled points where the student's classification "
        "differs from the reference's), smoothed with a trailing moving average - lower "
        "means the student more closely matches the reference"
    )

    linear_convergence_axes, log_convergence_axes = new_convergence_chart_pair(
        "convergence (linear scale)", "convergence (log scale)", float(training_set_size)
    )

    linear_convergence_axes.plot(n, disagreement)

    linear_convergence_axes.figure.canvas.manager.window.wm_geometry("+800+0")
    pyplot.show(block=False)

    print(
        "convergence chart (log scale): the same smoothed disagreement-rate series, "
        "log-scaled - useful for seeing how fast the student converges, since disagreement "
        "tends to drop roughly exponentially; the curve stops early once disagreement "
        "reaches exactly zero, which has no position on a log axis"
    )

    log_convergence_axes.plot(n, disagreement)

    log_convergence_axes.figure.canvas.manager.window.wm_geometry("+800+500")
    pyplot.show(block=False)

    # ---------------------

    print(
        "decision-boundary chart: training data points colored by class, the reference "
        "classifier's hyperplane(s) in green, and the trained student's in purple - the "
        "closer the purple lines are to the green ones, the more closely the student has "
        "learned the reference's decision boundary. The reference was chosen so its "
        "positive region (the intersection of its hyperplanes) is a bounded, closed shape, "
        "and the plot bounds are expanded as needed to keep that whole region visible"
    )

    plot_bounds = reference_region_bounds(reference_classifier, input_bounds)

    training_data_figure = new_figure("perceptrons (reference, student) with training data")
    axes = new_axes(training_data_figure, plot_bounds)

    plot_training_data(axes, training_data)
    plot_linear_classifier_network(axes, reference_classifier, color="green", x_bounds=plot_bounds[0])
    plot_linear_classifier_network(axes, student_classifier, color="purple", x_bounds=plot_bounds[0])

    pyplot.show()


if __name__ == "__main__":
    main()
