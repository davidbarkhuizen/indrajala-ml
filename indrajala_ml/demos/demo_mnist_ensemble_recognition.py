import os

import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot

from indrajala_ml.ensemble_train import train_ensemble_parallel_from_indices
from indrajala_ml.graphics.chart import (
    new_axes,
    new_confusion_matrix_figure,
    new_figure,
    sample_predictions_figure,
    style_dark_legend,
)
from indrajala_ml.model.fan_in_aware_backprop_classifier_network import FanInAwareBackpropClassifierNetwork
from indrajala_ml.mnist_data import (
    convert_parquet_to_binary,
    load_mnist_dataset,
    load_mnist_labels,
    load_mnist_records_at_indices,
)
from indrajala_ml.multiclass_evaluate import confusion_matrix

DIMENSION = 28 * 28
CLASS_COUNT = 10
LAYER_SIZES = [16]
IMAGE_SHAPE = (28, 28)

TRAIN_PARQUET = "data/mnist/mnist-train.parquet"
TEST_PARQUET = "data/mnist/mnist-test.parquet"
TRAIN_PATH = "data/mnist/mnist-train.bin"
TEST_PATH = "data/mnist/mnist-test.bin"
MODEL_PATH = "data/mnist/trained_model.json"


def _ensure_binary_files_exist() -> None:
    # a one-time, offline conversion (see mnist_data.convert_parquet_to_binary) - after this,
    # nothing in this demo (or in the actual training path) ever needs pyarrow again
    if not os.path.exists(TRAIN_PATH):
        print(f"converting {TRAIN_PARQUET} -> {TRAIN_PATH} (one-time)...")
        convert_parquet_to_binary(TRAIN_PARQUET, TRAIN_PATH)
    if not os.path.exists(TEST_PATH):
        print(f"converting {TEST_PARQUET} -> {TEST_PATH} (one-time)...")
        convert_parquet_to_binary(TEST_PARQUET, TEST_PATH)


def main() -> None:

    print(
        "Real MNIST (28x28, 60000 train / 10000 test) via EnsembleBackpropClassifierNetwork - "
        "10 completely independent FanInAwareBackpropClassifierNetworks, one per digit, each "
        "trained on its own class-balanced binary dataset with no state shared between them at "
        "all. That's what makes this genuinely (not just approximately) parallelizable across "
        "this machine's CPUs. Fan-in-aware initialization (rather than plain "
        "BackpropClassifierNetwork's default) measurably matters at this scale: +6.6 points "
        "test accuracy over the plain default."
    )
    print()

    _ensure_binary_files_exist()

    labels = load_mnist_labels(TRAIN_PATH)
    print(f"loaded {len(labels)} training labels (examples loaded lazily, per class, during training)")

    print("training all 10 digit classifiers (measured on this machine: ~30 minutes) ...")
    ensemble, diagnostics = train_ensemble_parallel_from_indices(
        TRAIN_PATH,
        load_mnist_records_at_indices,
        labels,
        class_count=CLASS_COUNT,
        layer_sizes=LAYER_SIZES,
        dimension=DIMENSION,
        input_bounds=[(0.0, 1.0)] * DIMENSION,
        learning_rate=0.5,
        epochs=5,
        classifier_cls=FanInAwareBackpropClassifierNetwork,
    )

    for label in range(CLASS_COUNT):
        diagnostic = diagnostics[label]
        status = diagnostic.status_label
        print(f"digit {label}: binary training accuracy {diagnostic.best_training_accuracy:.3f} ({status})")

    test_data = load_mnist_dataset(TEST_PATH)
    # accuracy from the confusion matrix's diagonal, rather than a second slow pure-Python pass
    # over the 10000 test examples
    matrix = confusion_matrix(ensemble, test_data, CLASS_COUNT)
    test_accuracy = sum(matrix[label][label] for label in range(CLASS_COUNT)) / len(test_data)
    print(f"overall 10-way test accuracy: {test_accuracy:.3f}")

    ensemble.save(MODEL_PATH)
    print(f"trained model saved to {MODEL_PATH} - reload it without retraining via:")
    print(f'  EnsembleBackpropClassifierNetwork.load("{MODEL_PATH}")')
    print()

    training_curves_figure = new_figure("MNIST recognition: per-digit training accuracy by epoch")
    training_curves_axes = new_axes(training_curves_figure, scaled=False)
    for label in range(CLASS_COUNT):
        accuracies = diagnostics[label].epoch_training_accuracies
        training_curves_axes.plot(range(1, len(accuracies) + 1), accuracies, label=f"digit {label}")
    training_curves_axes.set_xlabel("epoch")
    training_curves_axes.set_ylabel("binary training accuracy")
    style_dark_legend(training_curves_axes.legend(fontsize=8))

    new_confusion_matrix_figure("MNIST recognition: confusion matrix (test set)", matrix)

    sample_predictions_figure(
        "MNIST recognition: sample test predictions", test_data, ensemble.classify_state, image_shape=IMAGE_SHAPE
    )

    pyplot.show()


if __name__ == "__main__":
    main()
