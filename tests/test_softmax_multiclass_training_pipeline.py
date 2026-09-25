import random

from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.model.softmax_multiclass_backprop_classifier_network import (
    SoftmaxMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.multiclass_evaluate import accuracy
from indrajala_ml.train import train_linear_classifier_network


def test_train_linear_classifier_network_drives_softmax_multiclass_backprop_on_real_digit_data():

    # test_multiclass_training_pipeline.py's setup with softmax. Measured: training accuracy
    # 1.0 (converged), test 0.975, against one-vs-rest's 0.98125 and 0.9 on the same split (200
    # rows: not evidence softmax is better in general)
    random.seed(0)

    dataset = load_digits_dataset()
    subset = dataset[:200]
    train_data, test_data = split_train_test(subset, test_fraction=0.2, seed=1)

    student = SoftmaxMultiClassBackpropClassifierNetwork.randomized([16], 64, [(0.0, 1.0)] * 64, 10)
    result = train_linear_classifier_network(student, train_data, learning_rate=0.5, epochs=15)

    diagnostic = result.diagnostic
    assert diagnostic.best_training_accuracy == 1.0
    assert diagnostic.best_epoch_index == 14
    assert diagnostic.plateaued is False
    assert diagnostic.converged is True
    assert diagnostic.still_improving is False

    assert accuracy(student, test_data) == 0.975
