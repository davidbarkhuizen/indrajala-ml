import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.evaluate import agreement_label, class_balanced_disagreement_rate, compare_on_random_point
from indrajala_ml.geometry import square_bounds
from indrajala_ml.graphics.chart import (
    new_axes,
    new_figure,
    plot_classifier_probability_heatmap,
    plot_linear_classifier_network,
    plot_training_data,
)
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.train import reachable_reference_and_training_data, train_linear_classifier_network


def main() -> None:

    dimension = 2
    bounds = square_bounds(10.0, dimension)
    training_set_size = 1000

    # a single half-plane (cardinality=1), which both networks represent: a parity check, where
    # the other backprop demos pick targets the linear classifier can't learn
    reference, training_data = reachable_reference_and_training_data(1, dimension, bounds, training_set_size)

    print(
        "A single half-plane target - the easiest possible case, and squarely within "
        "LinearClassifierNetwork's own representational sweet spot. This demo isn't about "
        "which model is 'better' - it's a parity check, showing BackpropClassifierNetwork "
        "learns just as well as LinearClassifierNetwork here, not only on the harder targets "
        "the other backprop demos focus on."
    )
    print()

    linear_student = LinearClassifierNetwork.randomized(1, dimension, bounds)
    linear_result = train_linear_classifier_network(
        linear_student, training_data, learning_rate=0.25, epochs=5, reference_classifier=reference
    )

    backprop_student = BackpropClassifierNetwork.randomized([4], dimension, bounds)
    backprop_result = train_linear_classifier_network(
        backprop_student, training_data, learning_rate=1.0, epochs=20, reference_classifier=reference
    )

    for label, student, result in [("linear", linear_student, linear_result), ("backprop", backprop_student, backprop_result)]:
        diagnostic = result.diagnostic
        disagreement = class_balanced_disagreement_rate(reference, student, per_class_sample_count=300)
        new_state, reference_category, student_category = compare_on_random_point(reference, student)
        agreement = agreement_label(reference_category, student_category)
        print(
            f"{label}: training accuracy {diagnostic.best_training_accuracy:.3f}, disagreement "
            f"vs reference {disagreement:.3f}, prediction on new point: reference="
            f"{reference_category}, student={student_category} ({agreement})"
        )

    print()
    print(
        "chart: training data, the reference's straight boundary (green), the linear "
        "student's learned boundary (purple), and the backprop student's learned boundary as "
        "a probability heatmap underneath - on a target this simple, the heatmap's yellow/dark "
        "transition should track the green and purple lines closely."
    )

    figure = new_figure("backprop vs. linear: same target, same training data")
    axes = new_axes(figure, bounds)
    plot_classifier_probability_heatmap(axes, backprop_student, bounds)
    plot_training_data(axes, training_data)
    plot_linear_classifier_network(axes, reference, color="green", x_bounds=bounds[0])
    plot_linear_classifier_network(axes, linear_student, color="purple", x_bounds=bounds[0])
    pyplot.show()


if __name__ == "__main__":
    main()
