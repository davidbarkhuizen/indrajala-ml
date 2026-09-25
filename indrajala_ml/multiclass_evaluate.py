def confusion_matrix(
    classifier,
    test_data: list[tuple[tuple[float, ...], int]],
    class_count: int,
) -> list[list[int]]:
    """
    matrix[true_label][predicted_label] = count, over a fixed labeled test set.
    """

    matrix = [[0] * class_count for _ in range(class_count)]

    for state, true_label in test_data:
        predicted_label = classifier.classify_state(state)
        matrix[true_label][predicted_label] += 1

    return matrix


def accuracy(classifier, test_data: list[tuple[tuple[float, ...], int]]) -> float:
    """
    The fraction of test_data the classifier labels correctly: train.py's _training_accuracy, for a
    held-out test set.
    """

    assert len(test_data) >= 1, "test_data must not be empty"

    correct = sum(1 for state, true_label in test_data if classifier.classify_state(state) == true_label)
    return correct / len(test_data)
