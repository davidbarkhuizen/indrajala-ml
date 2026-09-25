import pytest

from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.multiclass_evaluate import accuracy, confusion_matrix
from tests.helpers import approx


def _fixed_network() -> MultiClassBackpropClassifierNetwork:
    # dimension=1, one hidden node, 2 output classes, fixed weights - deterministically
    # classifies state (2.0,) as class 0 (output node 0's activation, 0.599, exceeds output
    # node 1's, 0.544 - see tests/test_multiclass_backprop_model.py's identical fixture for the
    # hand-derived values)
    network = MultiClassBackpropClassifierNetwork([1], 1, [(-10.0, 10.0)], 2)
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node_0, output_node_1 = network.output_layer.nodes
    hidden_node.update_input_weights([0.5])
    hidden_node.bias = 0.1
    output_node_0.update_input_weights([0.8])
    output_node_0.bias = -0.2
    output_node_1.update_input_weights([-0.3])
    output_node_1.bias = 0.4
    return network


def test_accuracy_counts_correct_predictions():

    network = _fixed_network()
    # always predicts class 0 at this state - 2 correct, 1 wrong
    test_data = [((2.0,), 0), ((2.0,), 0), ((2.0,), 1)]

    assert accuracy(network, test_data) == approx(2 / 3)


def test_accuracy_rejects_empty_test_data():

    network = _fixed_network()

    with pytest.raises(AssertionError):
        accuracy(network, [])


def test_confusion_matrix_counts_true_vs_predicted_labels():

    network = _fixed_network()
    test_data = [((2.0,), 0), ((2.0,), 0), ((2.0,), 1)]

    matrix = confusion_matrix(network, test_data, class_count=2)

    assert matrix == [[2, 0], [1, 0]]


def test_confusion_matrix_diagonal_sums_to_correct_count():

    network = _fixed_network()
    test_data = [((2.0,), 0), ((2.0,), 0), ((2.0,), 1), ((2.0,), 0)]

    matrix = confusion_matrix(network, test_data, class_count=2)
    diagonal_sum = sum(matrix[i][i] for i in range(2))

    assert diagonal_sum == accuracy(network, test_data) * len(test_data)
