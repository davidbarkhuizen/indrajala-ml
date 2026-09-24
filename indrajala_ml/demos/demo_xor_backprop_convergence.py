import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.geometry import square_bounds
from indrajala_ml.graphics.chart import new_axes, new_figure, plot_classifier_probability_heatmap, plot_training_data
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.targets import XORTarget
from indrajala_ml.train import random_alternating_training_data, train_linear_classifier_network


def main() -> None:

    dimension = 2
    bounds = square_bounds(10.0, dimension)
    target = XORTarget(bounds)

    training_data = random_alternating_training_data(1000, target)

    print(
        "Same XOR-style target as demo_xor_linear_classifier_ceiling.py: category = (x > 0) != "
        "(y > 0), two diagonally opposite quadrants. That demo showed no LinearClassifierNetwork "
        "gate gets close, because its output layer's weights are fixed at 1.0 per hidden node - "
        "the output can only be a monotonically non-decreasing function of how many hidden nodes "
        "fire. BackpropClassifierNetwork's output layer is trained too, so a hidden node can "
        "push the output either way - this is what that difference buys."
    )
    print()

    student = BackpropClassifierNetwork.randomized([4], dimension, bounds)
    result = train_linear_classifier_network(student, training_data, learning_rate=1.0, epochs=150)

    diagnostic = result.diagnostic
    status = diagnostic.status_label
    print(
        f"training accuracy {diagnostic.best_training_accuracy:.3f} ({status} - best epoch "
        f"{diagnostic.best_epoch_index + 1}/{len(diagnostic.epoch_training_accuracies)}), well "
        "past the ~0.845 ceiling no LinearClassifierNetwork gate got past on this same target."
    )

    figure = new_figure("Backprop XOR: training data (true label) vs. learned probability heatmap")
    axes = new_axes(figure, bounds)
    plot_classifier_probability_heatmap(axes, student, bounds)
    plot_training_data(axes, training_data)
    pyplot.show()


if __name__ == "__main__":
    main()
