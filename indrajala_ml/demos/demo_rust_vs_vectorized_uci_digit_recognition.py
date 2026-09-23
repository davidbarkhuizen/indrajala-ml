import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.demos.timing import timed_train
from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.graphics.chart import new_axes, new_confusion_matrix_figure, new_figure, sample_predictions_figure
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.multiclass_evaluate import accuracy, confusion_matrix

DIMENSION = 64
CLASS_COUNT = 10
MODEL_PATH = "data/digits/trained_model_rust.json"


def main() -> None:

    print(
        "Rust vs numpy validation. The same UCI hand-written digits task "
        "demo_vectorized_uci_digit_recognition.py trains, extended with a third column: "
        "RustArrayMultiClassBackpropClassifierNetwork, the Rust-array-core-backed sibling that "
        "replaces numpy as the production backend (numpy stays on permanently as the benchmark "
        "comparison). Trains all three at the same seed/hyperparameters and reports each one's "
        "measured accuracy and wall-clock training time."
    )
    print()

    dataset = load_digits_dataset()
    train_data, test_data = split_train_test(dataset, test_fraction=0.2, seed=0)
    print(f"loaded {len(dataset)} samples: {len(train_data)} train, {len(test_data)} held out for testing")

    node_student = MultiClassBackpropClassifierNetwork.randomized(
        [32], DIMENSION, [(0.0, 1.0)] * DIMENSION, CLASS_COUNT
    )
    numpy_student = VectorizedMultiClassBackpropClassifierNetwork.randomized([32], DIMENSION, CLASS_COUNT)
    rust_student = RustArrayMultiClassBackpropClassifierNetwork.randomized([32], DIMENSION, CLASS_COUNT)

    node_result, node_elapsed = timed_train(node_student, train_data, learning_rate=0.5, epochs=30)
    numpy_result, numpy_elapsed = timed_train(numpy_student, train_data, learning_rate=0.5, epochs=30)
    rust_result, rust_elapsed = timed_train(rust_student, train_data, learning_rate=0.5, epochs=30)

    node_diagnostic = node_result.diagnostic
    numpy_diagnostic = numpy_result.diagnostic
    rust_diagnostic = rust_result.diagnostic
    node_test_accuracy = accuracy(node_student, test_data)
    numpy_test_accuracy = accuracy(numpy_student, test_data)
    rust_test_accuracy = accuracy(rust_student, test_data)

    print()
    print(
        f"pure-Python (MultiClassBackpropClassifierNetwork): training accuracy "
        f"{node_diagnostic.best_training_accuracy:.3f} ({node_diagnostic.status_label}), held-out "
        f"test accuracy {node_test_accuracy:.3f}, wall-clock {node_elapsed:.2f}s"
    )
    print(
        f"numpy (VectorizedMultiClassBackpropClassifierNetwork): training accuracy "
        f"{numpy_diagnostic.best_training_accuracy:.3f} ({numpy_diagnostic.status_label}), held-out "
        f"test accuracy {numpy_test_accuracy:.3f}, wall-clock {numpy_elapsed:.2f}s"
    )
    print(
        f"Rust (RustArrayMultiClassBackpropClassifierNetwork): training accuracy "
        f"{rust_diagnostic.best_training_accuracy:.3f} ({rust_diagnostic.status_label}), held-out "
        f"test accuracy {rust_test_accuracy:.3f}, wall-clock {rust_elapsed:.2f}s"
    )
    if numpy_elapsed > 0:
        print(f"Rust vs numpy speedup: {numpy_elapsed / rust_elapsed:.2f}x")
    if rust_elapsed > 0:
        print(f"Rust vs pure-Python speedup: {node_elapsed / rust_elapsed:.2f}x")

    rust_student.save(MODEL_PATH)
    print(f"trained Rust model saved to {MODEL_PATH} - reload it without retraining via:")
    print(f'  RustArrayMultiClassBackpropClassifierNetwork.load("{MODEL_PATH}")')
    print()

    matrix = confusion_matrix(rust_student, test_data, CLASS_COUNT)

    training_curve_figure = new_figure("Rust vs numpy digit recognition: training accuracy by epoch")
    training_curve_axes = new_axes(training_curve_figure, scaled=False)
    training_curve_axes.plot(
        range(1, len(node_diagnostic.epoch_training_accuracies) + 1),
        node_diagnostic.epoch_training_accuracies,
        label="pure-Python",
    )
    training_curve_axes.plot(
        range(1, len(numpy_diagnostic.epoch_training_accuracies) + 1),
        numpy_diagnostic.epoch_training_accuracies,
        label="numpy",
    )
    training_curve_axes.plot(
        range(1, len(rust_diagnostic.epoch_training_accuracies) + 1),
        rust_diagnostic.epoch_training_accuracies,
        label="Rust",
    )
    training_curve_axes.set_xlabel("epoch")
    training_curve_axes.set_ylabel("training accuracy")
    training_curve_axes.legend()

    new_confusion_matrix_figure("Rust digit recognition: confusion matrix (test set)", matrix)

    sample_predictions_figure(
        "Rust digit recognition: sample test predictions", test_data, rust_student.classify_state
    )

    pyplot.show()


if __name__ == "__main__":
    main()
