import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.graphics.chart import new_axes, new_confusion_matrix_figure, new_figure, sample_predictions_figure
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.multiclass_evaluate import accuracy, confusion_matrix
from indrajala_ml.train import train_linear_classifier_network

DIMENSION = 64
CLASS_COUNT = 10
MODEL_PATH = "data/digits/trained_model.json"


def main() -> None:

    print(
        "Classic-style handwritten digit recognition: the UCI ML hand-written digits dataset "
        "(8x8 pixel images, 10 classes, 1797 samples, bundled at data/digits/digits.csv), "
        "trained with MultiClassBackpropClassifierNetwork - a one-vs-rest multi-class sibling "
        "of BackpropClassifierNetwork, built on the same BackpropNode/BackpropLayer blocks."
    )
    print()

    dataset = load_digits_dataset()
    train_data, test_data = split_train_test(dataset, test_fraction=0.2, seed=0)
    print(f"loaded {len(dataset)} samples: {len(train_data)} train, {len(test_data)} held out for testing")

    student = MultiClassBackpropClassifierNetwork.randomized([32], DIMENSION, [(0.0, 1.0)] * DIMENSION, CLASS_COUNT)
    result = train_linear_classifier_network(student, train_data, learning_rate=0.5, epochs=30)

    diagnostic = result.diagnostic
    status = diagnostic.status_label
    test_accuracy = accuracy(student, test_data)
    print(
        f"training accuracy {diagnostic.best_training_accuracy:.3f} ({status} - best epoch "
        f"{diagnostic.best_epoch_index + 1}/{len(diagnostic.epoch_training_accuracies)}), "
        f"held-out test accuracy {test_accuracy:.3f}"
    )

    student.save(MODEL_PATH)
    print(f"trained model saved to {MODEL_PATH} - reload it without retraining via:")
    print(f'  MultiClassBackpropClassifierNetwork.load("{MODEL_PATH}")')
    print()

    matrix = confusion_matrix(student, test_data, CLASS_COUNT)

    training_curve_figure = new_figure("digit recognition: training accuracy by epoch")
    training_curve_axes = new_axes(training_curve_figure, scaled=False)
    training_curve_axes.plot(
        range(1, len(diagnostic.epoch_training_accuracies) + 1), diagnostic.epoch_training_accuracies
    )
    training_curve_axes.set_xlabel("epoch")
    training_curve_axes.set_ylabel("training accuracy")

    new_confusion_matrix_figure("digit recognition: confusion matrix (test set)", matrix)

    sample_predictions_figure("digit recognition: sample test predictions", test_data, student.classify_state)

    pyplot.show()


if __name__ == "__main__":
    main()
