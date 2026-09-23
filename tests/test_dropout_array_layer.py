from unittest.mock import patch

import numpy as np
import pytest

from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.dropout_array_layer import DropoutArrayLayer

# the same fixed weight/bias/input fixture test_dropout_layer.py's own DropoutNode suite uses -
# directly comparable numbers, not re-derived here: z=1.1, base=sigmoid(1.1)=0.7502601055951177
W = np.array([[0.5]])
B = np.array([0.1])
X = np.array([2.0])
BASE_ACTIVATION = np.array([0.7502601055951177])


def _dropout_layer(drop_probability: float = 0.5) -> DropoutArrayLayer:
    layer = DropoutArrayLayer(1, 1, drop_probability)
    layer.W = W.copy()
    layer.b = B.copy()
    return layer


def test_forward_at_eval_mode_matches_a_plain_sigmoid_no_rescale():

    # training defaults to False - the safe default if set_training_mode is never called
    layer = _dropout_layer()

    result = layer.forward(X)

    plain_layer = ArrayLayer(1, 1)
    plain_layer.W = W.copy()
    plain_layer.b = B.copy()
    expected = plain_layer.forward(X)

    assert np.allclose(result, expected)
    assert np.allclose(result, BASE_ACTIVATION)


def test_forward_in_training_mode_when_kept_rescales_by_one_over_keep_probability():

    layer = _dropout_layer(drop_probability=0.5)
    layer.set_training_mode(True)

    with patch("numpy.random.random", return_value=np.array([0.9])):  # 0.9 >= 0.5 -> kept
        result = layer.forward(X)

    assert np.allclose(result, BASE_ACTIVATION / 0.5)


def test_forward_in_training_mode_when_dropped_is_exactly_zero():

    layer = _dropout_layer(drop_probability=0.5)
    layer.set_training_mode(True)

    with patch("numpy.random.random", return_value=np.array([0.1])):  # 0.1 < 0.5 -> dropped
        result = layer.forward(X)

    assert result[0] == 0.0


def test_set_training_mode_false_reverts_to_eval_behavior():

    layer = _dropout_layer()
    layer.set_training_mode(True)
    with patch("numpy.random.random", return_value=np.array([0.1])):
        result = layer.forward(X)
    assert result[0] == 0.0  # dropped, while still in training mode

    layer.set_training_mode(False)
    result = layer.forward(X)

    assert np.allclose(result, BASE_ACTIVATION)


def test_compute_hidden_delta_when_kept_uses_the_unscaled_sigmoid_derivative():

    # the same subtlety that applies to DropoutNode: the derivative factor must be
    # base*(1-base) (the *pre*-scaling activation), not a*(1-a) on the rescaled activation
    layer = _dropout_layer(drop_probability=0.5)
    layer.set_training_mode(True)
    with patch("numpy.random.random", return_value=np.array([0.9])):  # kept
        layer.forward(X)

    next_layer = ArrayLayer(1, 1)
    next_layer.W = np.array([[0.8]])
    next_layer.delta = np.array([-0.5])

    layer.compute_hidden_delta(next_layer)

    sigmoid_derivative = BASE_ACTIVATION * (1.0 - BASE_ACTIVATION)
    expected = (-0.5 * 0.8) * sigmoid_derivative / 0.5
    assert np.allclose(layer.delta, expected)


def test_compute_hidden_delta_is_zero_when_the_unit_was_dropped():

    layer = _dropout_layer(drop_probability=0.5)
    layer.set_training_mode(True)
    with patch("numpy.random.random", return_value=np.array([0.1])):  # dropped
        layer.forward(X)

    next_layer = ArrayLayer(1, 1)
    next_layer.W = np.array([[0.8]])
    next_layer.delta = np.array([-0.5])

    layer.compute_hidden_delta(next_layer)

    assert layer.delta[0] == 0.0


def test_compute_hidden_delta_at_eval_mode_uses_no_rescale():

    layer = _dropout_layer()
    layer.forward(X)  # training defaults to False

    next_layer = ArrayLayer(1, 1)
    next_layer.W = np.array([[0.8]])
    next_layer.delta = np.array([-0.5])

    layer.compute_hidden_delta(next_layer)

    sigmoid_derivative = BASE_ACTIVATION * (1.0 - BASE_ACTIVATION)
    expected = (-0.5 * 0.8) * sigmoid_derivative
    assert np.allclose(layer.delta, expected)


def test_forward_batch_draws_an_independent_mask_per_row_not_one_shared_per_batch():

    layer = DropoutArrayLayer(4, 3, drop_probability=0.5)
    layer.W = np.random.default_rng(0).uniform(-1.0, 1.0, size=(4, 3))
    layer.b = np.random.default_rng(1).uniform(-1.0, 1.0, size=4)
    layer.set_training_mode(True)

    X_batch = np.random.default_rng(2).uniform(-1.0, 1.0, size=(20, 3))
    layer.forward_batch(X_batch)

    assert layer._mask_batch.shape == (20, 4)
    # not every row identical - an independent draw per example, not one mask for the batch
    assert len(set(tuple(row) for row in layer._mask_batch)) > 1


def test_forward_batch_at_eval_mode_matches_forward_per_row_stacked():

    layer = DropoutArrayLayer(3, 2, drop_probability=0.5)
    layer.W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.b = np.array([0.1, 0.2, 0.3])

    X_batch = np.array([[1.0, -1.0], [0.5, 0.5], [-2.0, 3.0]])
    expected_rows = [layer.forward(row).copy() for row in X_batch]

    result = layer.forward_batch(X_batch)

    assert np.allclose(result, np.stack(expected_rows))


def test_compute_hidden_delta_batch_at_eval_mode_matches_per_row_single_example_results():

    layer = DropoutArrayLayer(3, 2, drop_probability=0.5)
    layer.W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.b = np.array([0.1, 0.2, 0.3])

    next_layer = ArrayLayer(2, 3)
    next_layer.W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    next_layer.delta_batch = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.1]])

    X_batch = np.array([[1.0, -1.0], [0.5, 0.5], [-2.0, 3.0]])
    layer.forward_batch(X_batch)

    expected_rows = []
    for row_index in range(3):
        row_layer = DropoutArrayLayer(3, 2, drop_probability=0.5)
        row_layer.W = layer.W
        row_layer.b = layer.b
        row_layer.forward(X_batch[row_index])
        next_layer.delta = next_layer.delta_batch[row_index]
        row_layer.compute_hidden_delta(next_layer)
        expected_rows.append(row_layer.delta)

    layer.compute_hidden_delta_batch(next_layer)

    assert np.allclose(layer.delta_batch, np.stack(expected_rows))


def test_apply_accumulated_gradient_is_inherited_unchanged_from_array_layer():

    layer = _dropout_layer()
    layer.W = np.array([[1.0]])
    layer.b = np.array([5.0])
    layer.delta = np.array([1.0])
    layer.accumulate_gradient(np.array([1.0]))
    layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(layer.W, np.array([[0.9]]))
    assert np.allclose(layer.b, np.array([4.9]))


def test_drop_probability_of_one_is_rejected():

    with pytest.raises(AssertionError):
        DropoutArrayLayer(1, 1, drop_probability=1.0)
