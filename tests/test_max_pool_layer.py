import random

import pytest

from indrajala_ml.model.conv_layer import ConvLayer
from indrajala_ml.model.max_pool_layer import MaxPoolLayer
from indrajala_ml.model.state_layer import StateLayer
from tests.helpers import approx


def _pool_with_state(
    values: list[float], channels: int, height: int, width: int, pool_size: int, stride: int | None = None
) -> MaxPoolLayer:
    size = channels * height * width
    input_layer = StateLayer(size, [(-100.0, 100.0)] * size)
    input_layer.update_state(tuple(values))
    return MaxPoolLayer(
        input_layer=input_layer,
        input_height=height,
        input_width=width,
        input_channels=channels,
        pool_size=pool_size,
        stride=stride,
    )


def test_shape_defaults_to_non_overlapping_windows_and_keeps_channels():

    layer = _pool_with_state([0.0] * 72, channels=2, height=6, width=6, pool_size=2)

    assert layer.stride == 2
    assert (layer.out_height, layer.out_width, layer.channel_count) == (3, 3, 2)
    assert len(layer.nodes) == 2 * 3 * 3


def test_constructor_rejects_a_shape_mismatch_and_an_oversized_window():

    input_layer = StateLayer(16, [(-10.0, 10.0)] * 16)
    with pytest.raises(AssertionError):
        MaxPoolLayer(input_layer=input_layer, input_height=4, input_width=4, input_channels=2, pool_size=2)
    with pytest.raises(AssertionError):
        MaxPoolLayer(input_layer=input_layer, input_height=4, input_width=4, input_channels=1, pool_size=5)


def test_forward_matches_a_hand_computed_example_per_channel():

    # channel 0 (4x4, row-major):      channel 1 = -channel 0, so its maxima are each window's
    #   1  2 | 3  4                     *smallest* channel-0 value, negated:
    #   5  6 | 7  8                       -1 -3
    #   ---------------                   -9 -11
    #   9 10 |11 12
    #  13 14 |15 16                     channel 0's maxima: 6 8 / 14 16
    channel_0 = [float(v) for v in range(1, 17)]
    channel_1 = [-v for v in channel_0]
    layer = _pool_with_state(channel_0 + channel_1, channels=2, height=4, width=4, pool_size=2)

    layer.forward()

    assert [unit.value() for unit in layer.nodes] == [6.0, 8.0, 14.0, 16.0, -1.0, -3.0, -9.0, -11.0]
    # window slots are row-major within the window: 6 is slot 3 (bottom-right), -1 is slot 0
    assert layer.nodes[0].argmax_slot == 3
    assert layer.nodes[4].argmax_slot == 0


def test_ties_resolve_to_the_first_slot():

    layer = _pool_with_state([2.0, 2.0, 2.0, 2.0], channels=1, height=2, width=2, pool_size=2)
    layer.forward()

    assert layer.nodes[0].argmax_slot == 0


def test_downstream_sum_routes_each_delta_only_to_its_windows_argmax():

    channel_0 = [float(v) for v in range(1, 17)]
    layer = _pool_with_state(channel_0, channels=1, height=4, width=4, pool_size=2)
    layer.forward()
    for i, unit in enumerate(layer.nodes):
        unit.delta = 10.0 * (i + 1)

    # argmax input indices (row-major flat): 6 -> 5, 8 -> 7, 14 -> 13, 16 -> 15
    expected = {5: 10.0, 7: 20.0, 13: 30.0, 15: 40.0}
    for input_index in range(16):
        assert layer.downstream_sum(input_index) == expected.get(input_index, 0.0)


def test_overlapping_windows_send_every_winning_delta_to_a_shared_argmax():

    # 1x3 isn't square - use 3x3 with one dominant center pixel, pool_size=2, stride=1: all four
    # overlapping windows contain the center and all four pick it
    values = [0.0] * 9
    values[4] = 5.0
    layer = _pool_with_state(values, channels=1, height=3, width=3, pool_size=2, stride=1)
    layer.forward()
    for unit in layer.nodes:
        unit.delta = 1.5

    assert layer.downstream_sum(4) == approx(4 * 1.5)
    assert all(layer.downstream_sum(i) == 0.0 for i in range(9) if i != 4)


def test_weight_free_hooks_are_no_ops():

    layer = _pool_with_state([0.0] * 16, channels=1, height=4, width=4, pool_size=2)

    assert layer.snapshot_state() == []
    layer.restore_state([])
    layer.randomize_fan_in_aware()
    layer.accumulate_gradients()
    layer.apply_accumulated_gradients(0.1, batch_size=4)
    layer.apply_gradients(0.1)
    layer.set_training_mode(True)
    with pytest.raises(AssertionError):
        layer.restore_state([([1.0], 0.0)])


@pytest.mark.parametrize("pool_stride", [None, 1])
def test_gradient_check_through_conv_pool_conv(pool_stride: int | None):

    # finite differences through conv -> max pool -> conv, loss = sum of the last layer's
    # activations: the first conv layer's gradients have to flow back through the second conv
    # layer's downstream_sum *and* the pool layer's argmax routing. Random continuous inputs
    # make exact ties (where max isn't differentiable) vanishingly unlikely, and epsilon is far
    # smaller than any gap between window values here. Seed 5, not an arbitrary one: at seed 4
    # (stride 1) every last-layer kernel draws net-negative weights over the non-negative pooled
    # ReLU inputs, so every last-layer unit is dead and the check below would be vacuous.
    random.seed(5)
    side = 9
    input_layer = StateLayer(side * side, [(-10.0, 10.0)] * (side * side))
    input_layer.update_state(tuple(random.uniform(-2.0, 2.0) for _ in range(side * side)))
    first = ConvLayer(input_layer=input_layer, input_height=side, input_width=side, kernel_size=2, channel_count=2)
    pool = MaxPoolLayer(
        input_layer=first, input_height=8, input_width=8, input_channels=2, pool_size=2, stride=pool_stride
    )
    last = ConvLayer(
        input_layer=pool,
        input_height=pool.out_height,
        input_width=pool.out_width,
        kernel_size=2,
        channel_count=3,
        input_channels=2,
    )
    first.randomize_fan_in_aware()
    last.randomize_fan_in_aware()

    def total_loss() -> float:
        first.forward()
        pool.forward()
        last.forward()
        return sum(unit.value() for unit in last.nodes)

    total_loss()
    for unit in last.nodes:
        unit.delta = 1.0 if unit.value() > 0.0 else 0.0
    pool.compute_hidden_deltas(last)
    first.compute_hidden_deltas(pool)
    first.accumulate_gradients()
    last.accumulate_gradients()

    assert any(unit.delta != 0.0 for unit in first.nodes)

    epsilon = 1e-6
    for layer in (first, last):
        for kernel in layer.kernels:
            for i in range(len(kernel.weights)):
                original = kernel.weights[i]
                kernel.weights[i] = original + epsilon
                loss_plus = total_loss()
                kernel.weights[i] = original - epsilon
                loss_minus = total_loss()
                kernel.weights[i] = original
                assert kernel._weight_gradient_accum[i] == approx((loss_plus - loss_minus) / (2 * epsilon), abs=1e-5)
