from indrajala_ml.demos.timing import timed_train
from indrajala_ml.mnist_data import load_mnist_dataset
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.multiclass_evaluate import accuracy

DIMENSION = 28 * 28
CLASS_COUNT = 10
LAYER_SIZES = [30]  # matches demo_vectorized_mnist_recognition.py's own architecture, so this
# run's numbers are directly comparable to that demo's.
TRAIN_PATH = "data/mnist/mnist-train.bin"
TEST_PATH = "data/mnist/mnist-test.bin"


def main() -> None:

    print(
        "Rust vs numpy validation at real MNIST scale. One real training epoch over the full "
        "60000-example MNIST training set, same architecture/hyperparameters as "
        "demo_vectorized_mnist_recognition.py, now with a third network: "
        "RustArrayMultiClassBackpropClassifierNetwork, the Rust-array-core-backed sibling that "
        "replaces numpy as the production backend (numpy stays on permanently as the benchmark "
        "comparison)."
    )
    print()

    print(f"loading {TRAIN_PATH} / {TEST_PATH}...")
    train_data = load_mnist_dataset(TRAIN_PATH)
    test_data = load_mnist_dataset(TEST_PATH)
    print(f"loaded {len(train_data)} train / {len(test_data)} test examples")
    print()

    node_student = MultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, [(0.0, 1.0)] * DIMENSION, CLASS_COUNT
    )
    numpy_student = VectorizedMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    rust_student = RustArrayMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)

    print("training pure-Python network for 1 epoch (real baseline architecture - expect several minutes)...")
    node_result, node_elapsed = timed_train(node_student, train_data, learning_rate=0.5, epochs=1)
    node_test_accuracy = accuracy(node_student, test_data)
    print(f"  done in {node_elapsed:.1f}s ({node_elapsed / 60:.2f} min)")

    print("training numpy network for 1 epoch...")
    numpy_result, numpy_elapsed = timed_train(numpy_student, train_data, learning_rate=0.5, epochs=1)
    numpy_test_accuracy = accuracy(numpy_student, test_data)
    print(f"  done in {numpy_elapsed:.1f}s ({numpy_elapsed / 60:.2f} min)")

    print("training Rust network for 1 epoch...")
    rust_result, rust_elapsed = timed_train(rust_student, train_data, learning_rate=0.5, epochs=1)
    rust_test_accuracy = accuracy(rust_student, test_data)
    print(f"  done in {rust_elapsed:.1f}s ({rust_elapsed / 60:.2f} min)")

    print()
    print(
        f"pure-Python: training accuracy {node_result.diagnostic.best_training_accuracy:.3f}, "
        f"test accuracy {node_test_accuracy:.3f}, 1 epoch in {node_elapsed:.1f}s"
    )
    print(
        f"numpy:       training accuracy {numpy_result.diagnostic.best_training_accuracy:.3f}, "
        f"test accuracy {numpy_test_accuracy:.3f}, 1 epoch in {numpy_elapsed:.1f}s"
    )
    print(
        f"Rust:        training accuracy {rust_result.diagnostic.best_training_accuracy:.3f}, "
        f"test accuracy {rust_test_accuracy:.3f}, 1 epoch in {rust_elapsed:.1f}s"
    )
    if rust_elapsed > 0:
        print(f"Rust vs numpy speedup: {numpy_elapsed / rust_elapsed:.2f}x")
        print(f"Rust vs pure-Python speedup: {node_elapsed / rust_elapsed:.2f}x")
    print()


if __name__ == "__main__":
    main()
