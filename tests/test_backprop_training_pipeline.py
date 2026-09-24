import random

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.targets import XORTarget
from indrajala_ml.train import random_alternating_training_data, train_linear_classifier_network


def test_train_linear_classifier_network_drives_a_backprop_network_past_the_linear_ceiling_on_xor():

    # measured directly: best_training_accuracy=0.9666666666666667 at epoch 78/100 (plateaued,
    # not still improving - more epochs don't help further, since the handful of remaining
    # errors sit essentially on the x=0/y=0 boundary itself, which a bounded-weight sigmoid
    # network can only approximate, never perfectly resolve) - comfortably clear of the ~0.845
    # ceiling no LinearClassifierNetwork gate gets past on this same target (see test_train.py's
    # test_train_linear_classifier_network_keeps_the_best_epoch_not_the_last): the automated
    # counterpart to demo_xor_linear_classifier_ceiling.py.
    random.seed(0)

    bounds = square_bounds(10.0)
    target = XORTarget(bounds)
    training_data = random_alternating_training_data(300, target)

    student = BackpropClassifierNetwork.randomized([8], 2, bounds)
    result = train_linear_classifier_network(student, training_data, learning_rate=1.0, epochs=100)

    diagnostic = result.diagnostic
    assert diagnostic.best_training_accuracy == 0.9666666666666667
    assert diagnostic.best_epoch_index == 78
    assert diagnostic.plateaued is True
    assert diagnostic.converged is False
    assert diagnostic.still_improving is False

    # the trained student, not just the diagnostic's bookkeeping, actually predicts well
    correct = sum(1 for state, category in training_data if student.classify_state(state) == category)
    assert correct / len(training_data) == diagnostic.best_training_accuracy
