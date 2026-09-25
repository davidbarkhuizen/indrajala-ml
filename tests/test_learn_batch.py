import pytest

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.momentum_backprop_classifier_network import MomentumBackpropClassifierNetwork
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork


def test_learn_batch_of_one_matches_learn_exactly_for_backprop_classifier_network():

    state = (3.0, -4.0)
    category = 1.0
    learning_rate = 0.1

    via_learn = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    # rebuild via_learn_batch with the exact same starting weights as via_learn (randomized()
    # draws fresh random weights, so two independently-randomized networks would never match)
    via_learn_batch = BackpropClassifierNetwork([4], 2, square_bounds(10.0))
    via_learn_batch.restore(via_learn.snapshot())

    via_learn.learn(learning_rate, state, category)
    via_learn_batch.learn_batch(learning_rate, [(state, category)])

    assert via_learn.snapshot() == via_learn_batch.snapshot()


def test_learn_batch_of_one_matches_learn_exactly_for_multiclass_network():

    state = (3.0, -4.0)
    category = 2
    learning_rate = 0.1

    via_learn = MultiClassBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), class_count=3)
    via_learn_batch = MultiClassBackpropClassifierNetwork([4], 2, square_bounds(10.0), class_count=3)
    via_learn_batch.restore(via_learn.snapshot())

    via_learn.learn(learning_rate, state, category)
    via_learn_batch.learn_batch(learning_rate, [(state, category)])

    assert via_learn.snapshot() == via_learn_batch.snapshot()


def test_learn_batch_accumulates_over_every_example_before_updating_weights():

    # after only the first example of a 3-example batch, weights must be untouched - unlike
    # calling learn() three times in a row, which would update after each one
    learning_rate = 0.1
    batch = [((3.0, -4.0), 1.0), ((1.0, 2.0), 0.0), ((-5.0, 5.0), 1.0)]

    via_batch = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    starting_snapshot = via_batch.snapshot()

    via_batch._forward(batch[0][0])
    via_batch._backward(batch[0][1])
    via_batch._accumulate_gradients()

    assert via_batch.snapshot() == starting_snapshot


def test_learn_batch_matches_averaging_three_individual_learn_steps_gradients_by_hand():

    # a separate implementation: per-example gradients, averaged, applied once
    learning_rate = 0.1
    batch = [((3.0, -4.0), 1.0), ((1.0, 2.0), 0.0), ((-5.0, 5.0), 1.0)]

    via_learn_batch = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    via_manual = BackpropClassifierNetwork([4], 2, square_bounds(10.0))
    via_manual.restore(via_learn_batch.snapshot())

    via_learn_batch.learn_batch(learning_rate, batch)

    for state, category in batch:
        via_manual._forward(state)
        via_manual._backward(category)
        via_manual._accumulate_gradients()
    via_manual._apply_accumulated_gradients(learning_rate, len(batch))

    assert via_learn_batch.snapshot() == via_manual.snapshot()


def test_learn_batch_rejects_an_empty_batch():

    network = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    with pytest.raises(AssertionError):
        network.learn_batch(0.1, [])


def test_learn_batch_is_inherited_unchanged_by_momentum_sibling():

    # inherited learn_batch must reach the momentum node's apply_accumulated_gradient
    learning_rate = 0.1
    batch = [((3.0, -4.0), 1.0), ((1.0, 2.0), 0.0)]

    network = MomentumBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), momentum=0.5)
    before = network.snapshot()

    network.learn_batch(learning_rate, batch)

    after = network.snapshot()
    assert after != before  # weights actually moved
