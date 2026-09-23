import random

from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.multiclass_evaluate import accuracy
from indrajala_ml.train import train_linear_classifier_network


def test_train_linear_classifier_network_drives_multiclass_backprop_on_real_digit_data():

    # a small subset (200 of the 1797 bundled rows) and few epochs, so the test suite stays
    # fast - the full demo trains on the whole dataset. Measured directly (not guessed):
    # best_training_accuracy=0.98125 at epoch 13/15 (plateaued), test accuracy 0.9 on the
    # held-out split. Proves train_linear_classifier_network (written for the binary
    # BackpropClassifierNetwork) drives this multi-class sibling too, since it
    # only ever calls .learn()/.snapshot()/.restore()/.classify_state(), all of which
    # MultiClassBackpropClassifierNetwork implements with matching signatures.
    random.seed(0)

    dataset = load_digits_dataset()
    subset = dataset[:200]
    train_data, test_data = split_train_test(subset, test_fraction=0.2, seed=1)

    student = MultiClassBackpropClassifierNetwork.randomized([16], 64, [(0.0, 1.0)] * 64, 10)
    result = train_linear_classifier_network(student, train_data, learning_rate=0.5, epochs=15)

    diagnostic = result.diagnostic
    assert diagnostic.best_training_accuracy == 0.98125
    assert diagnostic.best_epoch_index == 12
    assert diagnostic.plateaued is True
    assert diagnostic.converged is False
    assert diagnostic.still_improving is False

    assert accuracy(student, test_data) == 0.9
