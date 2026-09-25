from indrajala_ml.demos.timing import timed_call, timed_train
from indrajala_ml.mnist_data import load_mnist_dataset, load_mnist_dataset_as_array
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.multiclass_evaluate import accuracy

DIMENSION = 28 * 28
CLASS_COUNT = 10
LAYER_SIZES = [30]  # the pure-Python baseline architecture this demo measures against
# (30-node hidden layer, ~12.5ms/iteration, ~12.5 minutes/epoch).
TRAIN_PATH = "data/mnist/mnist-train.bin"
TEST_PATH = "data/mnist/mnist-test.bin"


def main() -> None:

    print(
        "Vectorized vs pure-Python validation, real MNIST scale. One real training epoch over "
        "the full 60000-example MNIST training set, same architecture/hyperparameters, "
        "pure-Python MultiClassBackpropClassifierNetwork vs. its numpy-array-backed "
        "VectorizedMultiClassBackpropClassifierNetwork sibling (already parity-checked "
        "step-by-step in tests/test_vectorized_multiclass_backprop_model.py, and "
        "accuracy-trajectory-checked at UCI digits scale in "
        "demo_vectorized_uci_digit_recognition.py)."
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
    array_student = VectorizedMultiClassBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)

    print("training pure-Python network for 1 epoch (real baseline architecture - expect several minutes)...")
    node_result, node_elapsed = timed_train(node_student, train_data, learning_rate=0.5, epochs=1)
    node_test_accuracy = accuracy(node_student, test_data)
    print(f"  done in {node_elapsed:.1f}s ({node_elapsed / 60:.2f} min)")

    print("training vectorized network for 1 epoch...")
    array_result, array_elapsed = timed_train(array_student, train_data, learning_rate=0.5, epochs=1)
    array_test_accuracy = accuracy(array_student, test_data)
    print(f"  done in {array_elapsed:.1f}s ({array_elapsed / 60:.2f} min)")

    print()
    print(
        f"pure-Python: training accuracy {node_result.diagnostic.best_training_accuracy:.3f}, "
        f"test accuracy {node_test_accuracy:.3f}, 1 epoch in {node_elapsed:.1f}s"
    )
    print(
        f"vectorized:  training accuracy {array_result.diagnostic.best_training_accuracy:.3f}, "
        f"test accuracy {array_test_accuracy:.3f}, 1 epoch in {array_elapsed:.1f}s"
    )
    if array_elapsed > 0:
        print(f"speedup: {node_elapsed / array_elapsed:.2f}x")
    print()

    # a second measurement: the tuple decode's 47 million boxed floats against the array decode
    # (tests/test_mnist_data.py checks they agree)
    print("data loading comparison (full training file decode):")
    _, tuple_decode_elapsed = timed_call(load_mnist_dataset, TRAIN_PATH)
    _, array_decode_elapsed = timed_call(load_mnist_dataset_as_array, TRAIN_PATH)

    print(f"  load_mnist_dataset (tuples):        {tuple_decode_elapsed:.2f}s")
    print(f"  load_mnist_dataset_as_array (numpy): {array_decode_elapsed:.2f}s")
    if array_decode_elapsed > 0:
        print(f"  decode speedup: {tuple_decode_elapsed / array_decode_elapsed:.2f}x")


if __name__ == "__main__":
    main()
