import math
import random

import pytest

from indrajala_ml.model.conv_kernel import ConvKernel


def test_default_weights_are_zero_initialized_with_correct_fan_in():

    kernel = ConvKernel(kernel_size=3, in_channels=2)
    assert kernel.weights == [0.0] * (3 * 3 * 2)
    assert kernel.bias == 0.0


def test_explicit_weights_must_match_fan_in():

    ConvKernel(kernel_size=2, in_channels=1, weights=[0.1, 0.2, 0.3, 0.4])  # 2*2*1 = 4, fine

    with pytest.raises(AssertionError):
        ConvKernel(kernel_size=2, in_channels=1, weights=[0.1, 0.2, 0.3])  # only 3, needs 4


def test_randomize_fan_in_aware_draws_within_the_expected_limit():

    random.seed(0)
    kernel = ConvKernel(kernel_size=3, in_channels=1)  # fan_in = 9
    kernel.randomize_fan_in_aware()

    limit = 1.0 / math.sqrt(9)
    assert len(kernel.weights) == 9
    assert all(-limit <= w <= limit for w in kernel.weights)
    assert -limit <= kernel.bias <= limit
    # not literally all zero/identical - a real random draw happened
    assert len(set(kernel.weights)) > 1


def test_accumulate_then_apply_at_batch_size_one_matches_the_direct_formula():

    kernel = ConvKernel(kernel_size=1, in_channels=2, weights=[0.5, -0.5], bias=0.1)

    kernel.accumulate_gradient(delta=0.2, receptive_field_values=[1.0, 2.0])
    kernel.apply_accumulated_gradient(learning_rate=0.1, batch_size=1)

    # weight -= lr * delta * x, per position; bias -= lr * delta
    assert kernel.weights == pytest.approx([0.5 - 0.1 * 0.2 * 1.0, -0.5 - 0.1 * 0.2 * 2.0])
    assert kernel.bias == pytest.approx(0.1 - 0.1 * 0.2)


def test_accumulate_gradient_rejects_a_mismatched_receptive_field_length():

    kernel = ConvKernel(kernel_size=1, in_channels=2, weights=[0.5, -0.5])
    with pytest.raises(AssertionError):
        kernel.accumulate_gradient(delta=0.2, receptive_field_values=[1.0])


def test_multiple_spatial_positions_sum_not_average_while_batch_size_still_averages():

    # spatial positions are summed and only the batch is averaged: accumulate_gradient never
    # divides, apply_accumulated_gradient divides by batch_size
    kernel = ConvKernel(kernel_size=1, in_channels=1, weights=[0.5], bias=0.1)

    # "example 1": two spatial positions, values 1.0 and 2.0, same delta=0.2
    kernel.accumulate_gradient(delta=0.2, receptive_field_values=[1.0])
    kernel.accumulate_gradient(delta=0.2, receptive_field_values=[2.0])
    # "example 2": same two positions again
    kernel.accumulate_gradient(delta=0.2, receptive_field_values=[1.0])
    kernel.accumulate_gradient(delta=0.2, receptive_field_values=[2.0])

    # accum = 0.2*1.0 + 0.2*2.0 + 0.2*1.0 + 0.2*2.0 = 1.2, divided by batch_size 2, not by 4
    kernel.apply_accumulated_gradient(learning_rate=0.1, batch_size=2)
    assert kernel.weights == pytest.approx([0.5 - 0.1 * (1.2 / 2)])


def test_apply_accumulated_gradient_resets_the_accumulator():

    kernel = ConvKernel(kernel_size=1, in_channels=1, weights=[0.5], bias=0.1)

    kernel.accumulate_gradient(delta=0.2, receptive_field_values=[1.0])
    kernel.apply_accumulated_gradient(0.1, batch_size=1)
    weights_after_first_apply = list(kernel.weights)

    kernel.apply_accumulated_gradient(0.1, batch_size=1)
    assert kernel.weights == pytest.approx(weights_after_first_apply)
