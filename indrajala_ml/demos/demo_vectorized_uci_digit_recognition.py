import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.demos.timing import timed_train
from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.graphics.chart import new_axes, new_confusion_matrix_figure, new_figure, sample_predictions_figure
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.multiclass_evaluate import accuracy, confusion_matrix

DIMENSION = 64
CLASS_COUNT = 10
MODEL_PATH = "data/digits/trained_model_vectorized.json"


def main() -> None:

    print(
        "Vectorized vs pure-Python validation: the same UCI hand-written digits task "
        "demo_uci_digit_recognition.py trains, here trained by "
        "VectorizedMultiClassBackpropClassifierNetwork, a numpy-array-backed sibling, "
        "parity-checked step-by-step against the pure-Python MultiClassBackpropClassifierNetwork "
        "in tests/test_vectorized_multiclass_backprop_model.py. This demo trains both, at the "
        "same seed and hyperparameters, and reports the actual measured accuracy trajectory and "
        "wall-clock cost of each."
    )
    print()

    dataset = load_digits_dataset()
    train_data, test_data = split_train_test(dataset, test_fraction=0.2, seed=0)
    print(f"loaded {len(dataset)} samples: {len(train_data)} train, {len(test_data)} held out for testing")

    node_student = MultiClassBackpropClassifierNetwork.randomized(
        [32], DIMENSION, [(0.0, 1.0)] * DIMENSION, CLASS_COUNT
    )
    array_student = VectorizedMultiClassBackpropClassifierNetwork.randomized([32], DIMENSION, CLASS_COUNT)

    node_result, node_elapsed = timed_train(node_student, train_data, learning_rate=0.5, epochs=30)
    array_result, array_elapsed = timed_train(array_student, train_data, learning_rate=0.5, epochs=30)

    node_diagnostic = node_result.diagnostic
    array_diagnostic = array_result.diagnostic
    node_test_accuracy = accuracy(node_student, test_data)
    array_test_accuracy = accuracy(array_student, test_data)

    print()
    print(
        f"pure-Python (MultiClassBackpropClassifierNetwork): training accuracy "
        f"{node_diagnostic.best_training_accuracy:.3f} ({node_diagnostic.status_label}), held-out "
        f"test accuracy {node_test_accuracy:.3f}, wall-clock {node_elapsed:.2f}s"
    )
    print(
        f"vectorized (VectorizedMultiClassBackpropClassifierNetwork): training accuracy "
        f"{array_diagnostic.best_training_accuracy:.3f} ({array_diagnostic.status_label}), held-out "
        f"test accuracy {array_test_accuracy:.3f}, wall-clock {array_elapsed:.2f}s"
    )
    if array_elapsed > 0:
        print(f"speedup: {node_elapsed / array_elapsed:.2f}x")

    array_student.save(MODEL_PATH)
    print(f"trained vectorized model saved to {MODEL_PATH} - reload it without retraining via:")
    print(f'  VectorizedMultiClassBackpropClassifierNetwork.load("{MODEL_PATH}")')
    print()

    matrix = confusion_matrix(array_student, test_data, CLASS_COUNT)

    training_curve_figure = new_figure("vectorized digit recognition: training accuracy by epoch")
    training_curve_axes = new_axes(training_curve_figure, scaled=False)
    training_curve_axes.plot(
        range(1, len(node_diagnostic.epoch_training_accuracies) + 1),
        node_diagnostic.epoch_training_accuracies,
        label="pure-Python",
    )
    training_curve_axes.plot(
        range(1, len(array_diagnostic.epoch_training_accuracies) + 1),
        array_diagnostic.epoch_training_accuracies,
        label="vectorized",
    )
    training_curve_axes.set_xlabel("epoch")
    training_curve_axes.set_ylabel("training accuracy")
    training_curve_axes.legend()

    new_confusion_matrix_figure("vectorized digit recognition: confusion matrix (test set)", matrix)

    sample_predictions_figure(
        "vectorized digit recognition: sample test predictions", test_data, array_student.classify_state
    )

    pyplot.show()


if __name__ == "__main__":
    main()
