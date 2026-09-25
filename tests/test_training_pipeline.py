import random

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.axes import Axes

from indrajala_ml.evaluate import class_balanced_disagreement_rate
from indrajala_ml.geometry import square_bounds
from indrajala_ml.graphics.chart import (
    disagreement_axis_bounds,
    new_axes,
    new_figure,
    plot_linear_classifier_network,
    plot_training_data,
)
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.train import (
    random_alternating_training_data,
    reachable_reference_and_training_data,
    train_linear_classifier_network,
)


def test_training_of_linear_classifier():

    classifier_cardinality: int = 1
    dimension: int = 2
    l: float = 10.0

    learning_rate: float = 0.25
    training_set_size: int = 1000
    epoch_count: int = 1

    input_bounds = square_bounds(l, dimension)

    # generate a (random) reference classifier network and use it to produce a set of
    # training data
    #
    reference_classifier, training_data = reachable_reference_and_training_data(
        classifier_cardinality, dimension, input_bounds, training_set_size
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

    convergence_figure = new_figure("convergence")

    convergence_bounds = disagreement_axis_bounds(float(training_set_size))

    convergence_axes: Axes = new_axes(convergence_figure, convergence_bounds, scaled=False)

    n: list[int] = [x[0] for x in convergence_series]
    disagreement: list[float] = [x[1] for x in convergence_series]
    convergence_axes.plot(n, disagreement)  # pyright: ignore[reportUnknownMemberType]  (matplotlib's untyped **kwargs)

    # ---------------------

    training_data_figure = new_figure("perceptrons (reference, student) with training data")
    axes = new_axes(training_data_figure, student_classifier.input_bounds)

    axes.set_xlim(input_bounds[0])
    axes.set_ylim(input_bounds[1])

    plot_training_data(axes, training_data)
    plot_linear_classifier_network(axes, reference_classifier, color="green")
    plot_linear_classifier_network(axes, student_classifier, color="purple")

    pyplot.close("all")


def test_training_of_cardinality_two_linear_classifier_reduces_disagreement():

    random.seed(0)

    cardinality: int = 2
    dimension: int = 2
    l: float = 10.0
    bounds = square_bounds(l, dimension)

    reference = LinearClassifierNetwork.randomized(cardinality, dimension, bounds)
    training_data = random_alternating_training_data(400, reference)

    student = LinearClassifierNetwork.randomized(cardinality, dimension, bounds)

    disagreement_before = class_balanced_disagreement_rate(reference, student, per_class_sample_count=500)
    train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=5)
    disagreement_after = class_balanced_disagreement_rate(reference, student, per_class_sample_count=500)

    assert disagreement_after < disagreement_before
    assert disagreement_after < 0.15
