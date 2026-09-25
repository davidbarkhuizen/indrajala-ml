import random

import pytest

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.train import (
    _chunk_into_batches,
    reachable_reference_and_training_data,
    train_backprop_network_mini_batch,
    train_linear_classifier_network,
)


def test_chunk_into_batches_divides_evenly():
    batches = _chunk_into_batches(list(range(6)), batch_size=2)
    assert batches == [[0, 1], [2, 3], [4, 5]]


def test_chunk_into_batches_keeps_a_final_undersized_batch():
    batches = _chunk_into_batches(list(range(7)), batch_size=3)
    assert batches == [[0, 1, 2], [3, 4, 5], [6]]


def test_chunk_into_batches_of_size_one_matches_the_original_data():
    data = list(range(5))
    batches = _chunk_into_batches(data, batch_size=1)
    assert batches == [[0], [1], [2], [3], [4]]


def test_chunk_into_batches_larger_than_data_yields_a_single_batch():
    batches = _chunk_into_batches([1, 2, 3], batch_size=100)
    assert batches == [[1, 2, 3]]


def test_chunk_into_batches_rejects_batch_size_below_one():
    with pytest.raises(AssertionError):
        _chunk_into_batches([1, 2, 3], batch_size=0)


def test_train_mini_batch_rejects_empty_training_data():
    student = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    with pytest.raises(AssertionError):
        train_backprop_network_mini_batch(student, [], batch_size=4)


def test_batch_size_one_no_reshuffle_matches_train_linear_classifier_network_exactly():

    # batch_size=1 without reshuffling visits examples in train_linear_classifier_network's
    # order, so the snapshot and accuracy trajectory must be identical
    _reference, training_data = reachable_reference_and_training_data(1, 2, square_bounds(10.0), 200)

    via_learn = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    via_mini_batch = BackpropClassifierNetwork([4], 2, square_bounds(10.0))
    via_mini_batch.restore(via_learn.snapshot())

    learn_result = train_linear_classifier_network(via_learn, training_data, learning_rate=0.5, epochs=3)
    mini_batch_result = train_backprop_network_mini_batch(
        via_mini_batch, training_data, batch_size=1, learning_rate=0.5, epochs=3, reshuffle_each_epoch=False
    )

    assert via_learn.snapshot() == via_mini_batch.snapshot()
    assert learn_result.diagnostic.epoch_training_accuracies == mini_batch_result.diagnostic.epoch_training_accuracies


def test_larger_batch_size_still_trains_and_reports_epoch_accuracies():

    _reference, training_data = reachable_reference_and_training_data(1, 2, square_bounds(10.0), 200)
    student = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))
    before = student.snapshot()

    result = train_backprop_network_mini_batch(student, training_data, batch_size=16, learning_rate=0.5, epochs=5)

    assert student.snapshot() != before
    assert len(result.diagnostic.epoch_training_accuracies) == 5
    assert all(0.0 <= accuracy <= 1.0 for accuracy in result.diagnostic.epoch_training_accuracies)


def test_iterations_counts_batches_not_examples_when_tracking_a_reference_classifier():

    reference, training_data = reachable_reference_and_training_data(1, 2, square_bounds(10.0), 20)
    student = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))

    result = train_backprop_network_mini_batch(
        student, training_data, batch_size=4, learning_rate=0.5, epochs=1, reference_classifier=reference
    )

    # 20 examples / batch_size=4 = 5 batches in the one epoch, plus the initial pre-training
    # sample point
    assert len(result) == 1 + 5
    assert [iteration for iteration, _ in result] == [0, 1, 2, 3, 4, 5]


def test_train_mini_batch_calls_a_schedule_with_increasing_batch_step_indices():

    _reference, training_data = reachable_reference_and_training_data(1, 2, square_bounds(10.0), 20)
    student = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))

    calls: list[int] = []

    def recording_schedule(step: int) -> float:
        calls.append(step)
        return 0.5

    # 20 examples / batch_size=4 = 5 batches/epoch, 2 epochs = 10 calls total, not 20 - the
    # schedule steps against batches, not examples, per this function's own "iterations" note
    train_backprop_network_mini_batch(student, training_data, batch_size=4, learning_rate=recording_schedule, epochs=2)

    assert calls == list(range(10))


def test_train_mini_batch_works_with_multiclass_network_via_duck_typing():

    # one point per quadrant, seeded, with enough epochs to converge (pocket tracking means
    # "some weight moved" isn't guaranteed)
    random.seed(0)

    class_count = 3
    training_data = [
        ((1.0, 1.0), 0),
        ((-1.0, 1.0), 1),
        ((1.0, -1.0), 2),
        ((-1.0, -1.0), 0),
    ]
    student = MultiClassBackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0), class_count=class_count)

    result = train_backprop_network_mini_batch(student, training_data, batch_size=2, learning_rate=0.5, epochs=200)

    assert result.diagnostic.converged
