import math
import random

import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.geometry import square_bounds
from indrajala_ml.graphics.chart import new_axes, new_figure, plot_labeled_series
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.binary_cross_entropy_backprop_classifier_network import (
    BinaryCrossEntropyBackpropClassifierNetwork,
)
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.softmax_multiclass_backprop_classifier_network import (
    SoftmaxMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.multiclass_evaluate import accuracy as multiclass_accuracy
from indrajala_ml.train import random_alternating_training_data, train_linear_classifier_network

DIGITS_DIMENSION = 64
DIGITS_CLASS_COUNT = 10
DIGITS_LAYER_SIZES = [32]
DIGITS_EPOCHS = 30
DIGITS_LEARNING_RATE = 0.5

XOR_DIMENSION = 2
XOR_LAYER_SIZES = [8]
XOR_EPOCHS = 100


class XORTarget:
    # same target as demo_xor_linear_classifier_ceiling.py's XORTarget - kept as an independent
    # copy here (not imported), matching the same tests/ -> indrajala_ml/demos/ dependency
    # avoidance test_backprop_training_pipeline.py's own XORTarget copy already uses
    def __init__(self, bounds: list[tuple[float, float]]) -> None:
        self.input_bounds = bounds

    def classify_state(self, state: tuple[float, float]) -> float:
        x, y = state
        return 1.0 if (x > 0) != (y > 0) else 0.0


def _xavier_glorot_randomize(network) -> None:
    """
    Glorot & Bengio 2010's init (limit = sqrt(6/(fan_in+fan_out)) per layer, uniform) - measured
    not worth adopting for this codebase's shallow networks (a clean null against the fan-in-aware
    scheme MultiClassBackpropClassifierNetwork already uses). Kept only as a local comparison
    point in this demo, not a real library class, since it lost.
    """

    previous_size = network.dimension
    for layer in network.trainable_layers:
        limit = math.sqrt(6.0 / (previous_size + layer.size))
        for node in layer.nodes:
            node.update_input_weights([random.uniform(-limit, limit) for _ in range(previous_size)])
            node.bias = random.uniform(-limit, limit)
        previous_size = layer.size


def _compare_multiclass_loss_functions(train_data, test_data) -> list[tuple[str, str, list[int], list[float]]]:

    print("=== multi-class loss function: one-vs-rest (MSE) vs softmax (cross-entropy) ===")
    print(
        "Same architecture, same seed, same everything except the output layer/loss."
    )

    results = []
    for name, color, cls in [
        ("one-vs-rest (MSE)", "yellow", MultiClassBackpropClassifierNetwork),
        ("softmax (cross-entropy)", "cyan", SoftmaxMultiClassBackpropClassifierNetwork),
    ]:
        random.seed(0)
        student = cls.randomized(
            DIGITS_LAYER_SIZES, DIGITS_DIMENSION, [(0.0, 1.0)] * DIGITS_DIMENSION, DIGITS_CLASS_COUNT
        )
        result = train_linear_classifier_network(
            student, train_data, learning_rate=DIGITS_LEARNING_RATE, epochs=DIGITS_EPOCHS
        )
        diagnostic = result.diagnostic
        test_acc = multiclass_accuracy(student, test_data)
        print(
            f"  {name}: training accuracy {diagnostic.best_training_accuracy:.3f} (best epoch "
            f"{diagnostic.best_epoch_index + 1}/{DIGITS_EPOCHS}), held-out test accuracy {test_acc:.3f}"
        )
        epochs = list(range(1, len(diagnostic.epoch_training_accuracies) + 1))
        results.append((name, color, epochs, diagnostic.epoch_training_accuracies))
    print()

    return results


def _compare_binary_loss_functions() -> list[tuple[str, str, list[int], list[float]]]:

    print("=== binary loss function: quadratic (MSE) vs binary cross-entropy ===")
    print(
        "Same XOR scenario as test_backprop_training_pipeline.py's own pinned regression test. "
        "Cross-entropy needs a smaller learning rate than quadratic loss's own tuned value to "
        "perform comparably here - shown at both rates."
    )

    bounds = square_bounds(10.0, XOR_DIMENSION)
    target = XORTarget(bounds)

    configs = [
        ("quadratic (MSE), lr=1.0", "yellow", BackpropClassifierNetwork, 1.0),
        ("cross-entropy, lr=1.0 (untuned)", "orange", BinaryCrossEntropyBackpropClassifierNetwork, 1.0),
        ("cross-entropy, lr=0.1 (retuned)", "cyan", BinaryCrossEntropyBackpropClassifierNetwork, 0.1),
    ]

    results = []
    for name, color, cls, learning_rate in configs:
        # separate data-generation and weight-init seeds, matching the original investigation
        # script this section reproduces - not a single continuous draw sequence, so each
        # config's weight init doesn't depend on how many random calls generating training_data
        # happened to consume first
        random.seed(0)
        training_data = random_alternating_training_data(300, target)
        random.seed(1000)
        student = cls.randomized(XOR_LAYER_SIZES, XOR_DIMENSION, bounds)
        result = train_linear_classifier_network(
            student, training_data, learning_rate=learning_rate, epochs=XOR_EPOCHS
        )
        diagnostic = result.diagnostic
        print(
            f"  {name}: training accuracy {diagnostic.best_training_accuracy:.3f} (best epoch "
            f"{diagnostic.best_epoch_index + 1}/{XOR_EPOCHS})"
        )
        epochs = list(range(1, len(diagnostic.epoch_training_accuracies) + 1))
        results.append((name, color, epochs, diagnostic.epoch_training_accuracies))
    print()

    return results


def _compare_init_schemes(train_data, test_data) -> list[tuple[str, str, list[int], list[float]]]:

    print("=== init scheme: fan-in-aware vs Xavier/Glorot ===")
    print(
        "Same architecture/data as the multi-class loss comparison above, on the UCI digits set."
    )

    results = []
    for name, color, randomize_fn in [
        ("fan-in-aware (production default)", "yellow", None),
        ("Xavier/Glorot", "magenta", _xavier_glorot_randomize),
    ]:
        random.seed(0)
        student = MultiClassBackpropClassifierNetwork(
            DIGITS_LAYER_SIZES, DIGITS_DIMENSION, [(0.0, 1.0)] * DIGITS_DIMENSION, DIGITS_CLASS_COUNT
        )
        if randomize_fn is None:
            student.randomize()
        else:
            randomize_fn(student)
        result = train_linear_classifier_network(
            student, train_data, learning_rate=DIGITS_LEARNING_RATE, epochs=DIGITS_EPOCHS
        )
        diagnostic = result.diagnostic
        test_acc = multiclass_accuracy(student, test_data)
        print(
            f"  {name}: training accuracy {diagnostic.best_training_accuracy:.3f} (best epoch "
            f"{diagnostic.best_epoch_index + 1}/{DIGITS_EPOCHS}), held-out test accuracy {test_acc:.3f}"
        )
        epochs = list(range(1, len(diagnostic.epoch_training_accuracies) + 1))
        results.append((name, color, epochs, diagnostic.epoch_training_accuracies))
    print()

    return results


def main() -> None:

    print(
        "Runs three comparisons: multi-class loss function (one-vs-rest vs softmax), binary loss "
        "function (quadratic vs cross-entropy), and weight-init scheme (fan-in-aware vs "
        "Xavier/Glorot). Prints each comparison's measured numbers and plots a "
        "training-accuracy-by-epoch chart per section."
    )
    print()

    dataset = load_digits_dataset()
    train_data, test_data = split_train_test(dataset, test_fraction=0.2, seed=0)
    print(f"UCI digits: {len(dataset)} samples, {len(train_data)} train / {len(test_data)} test")
    print()

    multiclass_loss_results = _compare_multiclass_loss_functions(train_data, test_data)
    binary_loss_results = _compare_binary_loss_functions()
    init_scheme_results = _compare_init_schemes(train_data, test_data)

    for title, results in [
        ("multi-class loss function: training accuracy by epoch (UCI digits)", multiclass_loss_results),
        ("binary loss function: training accuracy by epoch (XOR)", binary_loss_results),
        ("init scheme: training accuracy by epoch (UCI digits)", init_scheme_results),
    ]:
        figure = new_figure(title)
        axes = new_axes(figure, scaled=False)
        plot_labeled_series(axes, results)
        axes.set_xlabel("epoch")
        axes.set_ylabel("training accuracy")

    pyplot.show()


if __name__ == "__main__":
    main()
