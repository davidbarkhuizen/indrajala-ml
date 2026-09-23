import numpy as np
import pytest

import indrajala_ml_array as pa
from indrajala_ml.model.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer

# the same fixed weight/bias/input fixture test_dropout_array_layer.py's own DropoutArrayLayer
# suite uses - directly comparable numbers: z=1.1, base=sigmoid(1.1)=0.7502601055951177
W = [[0.5]]
B = [0.1]
X = [2.0]
BASE_ACTIVATION = 0.7502601055951177


def _dropout_layer(drop_probability: float = 0.5) -> DropoutRustArrayLayer:
    layer = DropoutRustArrayLayer(1, 1, drop_probability)
    layer.W = pa.Array(W)
    layer.b = pa.Array(B)
    return layer


def test_forward_at_eval_mode_matches_a_plain_sigmoid_no_rescale():

    layer = _dropout_layer()

    result = layer.forward(pa.Array(X))

    plain_layer = RustArrayLayer(1, 1)
    plain_layer.W = pa.Array(W)
    plain_layer.b = pa.Array(B)
    expected = plain_layer.forward(pa.Array(X))

    assert np.allclose(result.tolist(), expected.tolist())
    assert result[0] == pytest.approx(BASE_ACTIVATION)


def test_forward_and_hidden_delta_in_training_mode_are_internally_consistent_across_many_draws():

    # a bit-identical mask draw against the numpy-backed sibling isn't achievable (this crate's
    # hand-rolled xorshift128+ generator can never reproduce numpy's Mersenne Twister stream,
    # the same RNG-incomparability that applies to uniform()) - so this checks the fused op's
    # own internal contract instead: every kept unit's activation equals
    # base/keep_probability exactly, every dropped unit's activation is exactly 0.0, and the
    # returned mask is what forward_batch actually used, not re-derived - across enough draws
    # that both outcomes are certain to appear at drop_probability=0.5
    layer = _dropout_layer(drop_probability=0.5)
    layer.set_training_mode(True)

    saw_kept = False
    saw_dropped = False
    for _ in range(200):
        result = layer.forward(pa.Array(X))
        mask_value = layer._mask[0]
        assert mask_value in (0.0, 1.0)
        if mask_value == 1.0:
            saw_kept = True
            assert result[0] == pytest.approx(BASE_ACTIVATION / 0.5)
        else:
            saw_dropped = True
            assert result[0] == 0.0
        assert layer._base_activation[0] == pytest.approx(BASE_ACTIVATION)

    assert saw_kept and saw_dropped


def test_compute_hidden_delta_when_kept_uses_the_unscaled_sigmoid_derivative():

    layer = _dropout_layer(drop_probability=0.5)
    layer.set_training_mode(True)

    # force a kept outcome by retrying until one occurs - the fused op has no seam to inject a
    # forced draw through (unlike numpy's own patch("numpy.random.random", ...)), so this relies
    # on drop_probability=0.5 making a kept draw certain within a handful of attempts
    for _ in range(200):
        layer.forward(pa.Array(X))
        if layer._mask[0] == 1.0:
            break
    else:
        pytest.fail("never drew a kept outcome in 200 attempts")

    next_layer = RustArrayLayer(1, 1)
    next_layer.W = pa.Array([[0.8]])
    next_layer.delta = pa.Array([-0.5])

    layer.compute_hidden_delta(next_layer)

    sigmoid_derivative = BASE_ACTIVATION * (1.0 - BASE_ACTIVATION)
    expected = (-0.5 * 0.8) * sigmoid_derivative / 0.5
    assert layer.delta[0] == pytest.approx(expected)


def test_compute_hidden_delta_is_zero_when_the_unit_was_dropped():

    layer = _dropout_layer(drop_probability=0.5)
    layer.set_training_mode(True)

    for _ in range(200):
        layer.forward(pa.Array(X))
        if layer._mask[0] == 0.0:
            break
    else:
        pytest.fail("never drew a dropped outcome in 200 attempts")

    next_layer = RustArrayLayer(1, 1)
    next_layer.W = pa.Array([[0.8]])
    next_layer.delta = pa.Array([-0.5])

    layer.compute_hidden_delta(next_layer)

    assert layer.delta[0] == 0.0


def test_compute_hidden_delta_at_eval_mode_uses_no_rescale():

    layer = _dropout_layer()
    layer.forward(pa.Array(X))  # training defaults to False

    next_layer = RustArrayLayer(1, 1)
    next_layer.W = pa.Array([[0.8]])
    next_layer.delta = pa.Array([-0.5])

    layer.compute_hidden_delta(next_layer)

    sigmoid_derivative = BASE_ACTIVATION * (1.0 - BASE_ACTIVATION)
    expected = (-0.5 * 0.8) * sigmoid_derivative
    assert layer.delta[0] == pytest.approx(expected)


def test_forward_batch_draws_an_independent_mask_per_row_not_one_shared_per_batch():

    layer = DropoutRustArrayLayer(4, 3, drop_probability=0.5)
    layer.W = pa.uniform(-1.0, 1.0, (4, 3))
    layer.b = pa.uniform(-1.0, 1.0, 4)
    layer.set_training_mode(True)

    X_batch = pa.uniform(-1.0, 1.0, (20, 3))
    layer.forward_batch(X_batch)

    assert layer._mask_batch.shape == (20, 4)
    rows = layer._mask_batch.tolist()
    assert len(set(tuple(row) for row in rows)) > 1


def test_forward_batch_at_eval_mode_matches_forward_per_row_stacked():

    layer = DropoutRustArrayLayer(3, 2, drop_probability=0.5)
    layer.W = pa.Array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.b = pa.Array([0.1, 0.2, 0.3])

    x_rows = [[1.0, -1.0], [0.5, 0.5], [-2.0, 3.0]]
    expected_rows = [layer.forward(pa.Array(row)).tolist() for row in x_rows]

    result = layer.forward_batch(pa.Array(x_rows))

    assert np.allclose(result.tolist(), expected_rows)


def test_compute_output_delta_and_apply_accumulated_gradient_are_inherited_unchanged():

    layer = _dropout_layer()
    layer.W = pa.Array([[1.0]])
    layer.b = pa.Array([5.0])
    layer.delta = pa.Array([1.0])
    layer.accumulate_gradient(pa.Array([1.0]))
    layer.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    assert np.allclose(layer.W.tolist(), [[0.9]])
    assert np.allclose(layer.b.tolist(), [4.9])


def test_drop_probability_of_one_is_rejected():

    with pytest.raises(AssertionError):
        DropoutRustArrayLayer(1, 1, drop_probability=1.0)
