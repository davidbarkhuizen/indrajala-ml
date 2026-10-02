from indrajala_ml.data.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.model.networks.python.softmax_multiclass_backprop_classifier_network import (
    SoftmaxMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.training.multiclass_evaluate import accuracy
from indrajala_ml.training.train import train_linear_classifier_network


def test_train_linear_classifier_network_drives_softmax_multiclass_backprop_on_real_digit_data():

    # test_multiclass_training_pipeline.py's setup with softmax. Measured from seed 0: training
    # accuracy 0.9875 (still improving), test 0.925, against one-vs-rest's 0.975 and 0.95 on the
    # same split and seed (200 rows: no evidence either loss is better in general)

    dataset = load_digits_dataset()
    subset = dataset[:200]
    train_data, test_data = split_train_test(subset, test_fraction=0.2, seed=1)

    student = SoftmaxMultiClassBackpropClassifierNetwork.randomized([16], 64, [(0.0, 1.0)] * 64, 10, seed=0)
    result = train_linear_classifier_network(student, train_data, learning_rate=0.5, epochs=15)

    diagnostic = result.diagnostic
    assert diagnostic.best_training_accuracy == 0.9875
    assert diagnostic.best_epoch_index == 14
    assert diagnostic.plateaued is False
    assert diagnostic.converged is False
    assert diagnostic.still_improving is True

    assert accuracy(student, test_data) == 0.925
