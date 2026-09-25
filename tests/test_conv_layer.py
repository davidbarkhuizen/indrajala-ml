import random
from collections.abc import Sequence

import pytest

from indrajala_ml.model.conv_layer import ConvLayer
from indrajala_ml.model.state_layer import StateLayer


def _layer_with_state(
    values: list[float], height: int, width: int, kernel_size: int, channel_count: int = 1, stride: int = 1
) -> ConvLayer:
    input_layer = StateLayer(height * width, [(-10.0, 10.0)] * (height * width))
    input_layer.update_state(tuple(values))
    return ConvLayer(
        input_layer=input_layer,
        input_height=height,
        input_width=width,
        kernel_size=kernel_size,
        channel_count=channel_count,
        stride=stride,
    )


def test_output_shape_and_valid_padding():

    layer = _layer_with_state([0.0] * 16, height=4, width=4, kernel_size=3, channel_count=2)

    assert layer.out_height == 2  # (4-3)//1 + 1
    assert layer.out_width == 2
    assert len(layer.nodes) == 2 * 2 * 2  # channel_count * out_height * out_width


def test_nodes_are_channel_major_sharing_one_kernel_per_channel():

    layer = _layer_with_state([0.0] * 16, height=4, width=4, kernel_size=3, channel_count=2)

    positions_per_channel = layer.out_height * layer.out_width
    first_channel_units = layer.nodes[:positions_per_channel]
    second_channel_units = layer.nodes[positions_per_channel:]

    assert all(unit.kernel is layer.kernels[0] for unit in first_channel_units)
    assert all(unit.kernel is layer.kernels[1] for unit in second_channel_units)
    assert layer.kernels[0] is not layer.kernels[1]


def test_constructor_rejects_a_kernel_larger_than_the_input():

    input_layer = StateLayer(9, [(-10.0, 10.0)] * 9)
    with pytest.raises(AssertionError):
        ConvLayer(input_layer=input_layer, input_height=3, input_width=3, kernel_size=4, channel_count=1)


def test_constructor_rejects_a_shape_mismatch_with_input_layer():

    input_layer = StateLayer(9, [(-10.0, 10.0)] * 9)  # 9 nodes
    with pytest.raises(AssertionError):
        ConvLayer(input_layer=input_layer, input_height=4, input_width=4, kernel_size=2, channel_count=1)  # 16 != 9


def test_forward_matches_a_hand_computed_small_example():

    # 3x3 input [[1,2,3],[4,5,6],[7,8,9]], kernel_size=2, stride=1 -> 2x2 output. Weights
    # [1,0,0,0] in (kr, kc) order pick each window's top-left element; hand-derived:
    #   (0,0): window [1,2,4,5] . [1,0,0,0] = 1 -> relu(1) = 1
    #   (0,1): window [2,3,5,6] . [1,0,0,0] = 2 -> relu(2) = 2
    #   (1,0): window [4,5,7,8] . [1,0,0,0] = 4 -> relu(4) = 4
    #   (1,1): window [5,6,8,9] . [1,0,0,0] = 5 -> relu(5) = 5
    layer = _layer_with_state([1, 2, 3, 4, 5, 6, 7, 8, 9], height=3, width=3, kernel_size=2)
    layer.kernels[0].weights = [1.0, 0.0, 0.0, 0.0]
    layer.kernels[0].bias = 0.0

    layer.forward()

    assert [unit.value() for unit in layer.nodes] == pytest.approx([1.0, 2.0, 4.0, 5.0])


def test_receptive_field_wiring_via_a_single_hot_pixel():

    # 4x4 input with one hot pixel at (1, 2), all-ones 2x2 kernel, zero bias: exactly the four
    # windows covering (1, 2) - outputs (0,1), (0,2), (1,1), (1,2) of the 3x3 grid - are 1.0
    height = width = 4
    values = [0.0] * (height * width)
    hot_row, hot_col = 1, 2
    values[hot_row * width + hot_col] = 1.0

    layer = _layer_with_state(values, height=height, width=width, kernel_size=2)
    layer.kernels[0].weights = [1.0, 1.0, 1.0, 1.0]
    layer.kernels[0].bias = 0.0
    layer.forward()

    assert layer.out_height == 3 and layer.out_width == 3
    expected_nonzero = {(0, 1), (0, 2), (1, 1), (1, 2)}
    for row in range(3):
        for col in range(3):
            unit = layer.nodes[row * 3 + col]
            expected = 1.0 if (row, col) in expected_nonzero else 0.0
            assert unit.value() == pytest.approx(expected), f"position ({row},{col})"


def _multichannel_layer_with_state(
    values: Sequence[float], channels: int, height: int, width: int, kernel_size: int, channel_count: int = 1
) -> ConvLayer:
    size = channels * height * width
    input_layer = StateLayer(size, [(-100.0, 100.0)] * size)
    input_layer.update_state(tuple(values))
    return ConvLayer(
        input_layer=input_layer,
        input_height=height,
        input_width=width,
        kernel_size=kernel_size,
        channel_count=channel_count,
        input_channels=channels,
    )


def test_multichannel_kernels_span_every_input_channel():

    layer = _multichannel_layer_with_state([0.0] * 32, channels=2, height=4, width=4, kernel_size=3, channel_count=5)

    assert all(len(kernel.weights) == 2 * 3 * 3 for kernel in layer.kernels)
    assert all(len(unit.input_nodes) == 2 * 3 * 3 for unit in layer.nodes)
    assert len(layer.nodes) == 5 * 2 * 2  # channel_count * out_height * out_width


def test_constructor_rejects_an_input_channels_mismatch_with_input_layer():

    input_layer = StateLayer(18, [(-10.0, 10.0)] * 18)  # 2 channels of 3x3
    with pytest.raises(AssertionError):
        ConvLayer(
            input_layer=input_layer, input_height=3, input_width=3, kernel_size=2, channel_count=1, input_channels=3
        )


def test_multichannel_forward_matches_a_hand_computed_small_example():

    # two 3x3 input channels, channel-major: channel 0 = [[1,2,3],[4,5,6],[7,8,9]], channel 1 =
    # 10x that. kernel_size=2, one output channel; weights in (channel, kr, kc) order are
    # [1,0,0,0 | 0,0,0,1] - channel 0's window top-left plus channel 1's window bottom-right:
    #   (0,0): 1 + 50 = 51    (0,1): 2 + 60 = 62
    #   (1,0): 4 + 80 = 84    (1,1): 5 + 90 = 95
    channel_0 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    channel_1 = [10 * v for v in channel_0]
    layer = _multichannel_layer_with_state(channel_0 + channel_1, channels=2, height=3, width=3, kernel_size=2)
    layer.kernels[0].weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    layer.kernels[0].bias = 0.0

    layer.forward()

    assert [unit.value() for unit in layer.nodes] == pytest.approx([51.0, 62.0, 84.0, 95.0])


@pytest.mark.parametrize("hot_channel", [0, 1, 2])
def test_multichannel_receptive_field_wiring_via_a_single_hot_pixel(hot_channel):

    # each input channel's kernel slice is a distinct constant (1, 2, 3), so the single-channel
    # test's positions must carry the hot channel's constant; a wrong channel offset changes it
    height = width = 4
    plane = height * width
    values = [0.0] * (3 * plane)
    values[hot_channel * plane + 1 * width + 2] = 1.0

    layer = _multichannel_layer_with_state(values, channels=3, height=height, width=width, kernel_size=2)
    layer.kernels[0].weights = [1.0] * 4 + [2.0] * 4 + [3.0] * 4
    layer.kernels[0].bias = 0.0
    layer.forward()

    expected_nonzero = {(0, 1), (0, 2), (1, 1), (1, 2)}
    for row in range(3):
        for col in range(3):
            expected = float(hot_channel + 1) if (row, col) in expected_nonzero else 0.0
            assert layer.nodes[row * 3 + col].value() == pytest.approx(expected), f"position ({row},{col})"


def test_a_conv_layer_feeds_the_next_directly_as_channel_major_input():

    # .nodes is channel-major, the order the next layer reads: with 1x1 kernels, second-layer
    # unit (row, col) reads first-layer channel c at (row, col) as its c-th input
    input_layer = StateLayer(9, [(-10.0, 10.0)] * 9)
    first = ConvLayer(input_layer=input_layer, input_height=3, input_width=3, kernel_size=2, channel_count=2)
    second = ConvLayer(
        input_layer=first, input_height=2, input_width=2, kernel_size=1, channel_count=1, input_channels=2
    )

    for position, unit in enumerate(second.nodes):
        assert unit.input_nodes[0] is first.nodes[position]
        assert unit.input_nodes[1] is first.nodes[4 + position]


def test_apply_gradients_matches_a_hand_computed_single_example():

    layer = _layer_with_state([1, 2, 3, 4, 5, 6, 7, 8, 9], height=3, width=3, kernel_size=2)
    layer.kernels[0].weights = [1.0, 0.0, 0.0, 0.0]
    layer.kernels[0].bias = 0.0
    layer.forward()

    for unit in layer.nodes:
        unit.delta = 0.1  # a fixed downstream delta at every position, for a simple hand check

    # accum[i] = sum over positions of delta * window[i]; windows [1,2,4,5], [2,3,5,6],
    # [4,5,7,8], [5,6,8,9], so index 0 sums 1+2+4+5 = 12
    layer.apply_gradients(learning_rate=0.1)

    expected_weight_0 = 1.0 - 0.1 * (0.1 * 1 + 0.1 * 2 + 0.1 * 4 + 0.1 * 5)
    assert layer.kernels[0].weights[0] == pytest.approx(expected_weight_0)
    expected_bias = 0.0 - 0.1 * (0.1 * 4)  # 4 positions, delta=0.1 each, summed then batch_size=1
    assert layer.kernels[0].bias == pytest.approx(expected_bias)


def test_apply_accumulated_gradients_applies_once_per_kernel_not_once_per_unit():

    # several units share each kernel; the kernel must move once, by the summed gradient over
    # batch_size
    layer = _layer_with_state([1.0] * 16, height=4, width=4, kernel_size=3, channel_count=2)
    layer.forward()
    for unit in layer.nodes:
        unit.delta = 0.5
    layer.accumulate_gradients()

    positions_per_channel = layer.out_height * layer.out_width  # 4
    expected_accum_per_weight = 0.5 * 1.0 * positions_per_channel  # every input value is 1.0

    for kernel in layer.kernels:
        assert kernel._weight_gradient_accum == pytest.approx([expected_accum_per_weight] * len(kernel.weights))

    layer.apply_accumulated_gradients(learning_rate=0.1, batch_size=2)

    expected_weight = 0.0 - 0.1 * (expected_accum_per_weight / 2)
    for kernel in layer.kernels:
        assert kernel.weights == pytest.approx([expected_weight] * len(kernel.weights))
        # accumulator reset - a second apply with nothing newly accumulated must be a no-op
    weights_after_first_apply = [list(k.weights) for k in layer.kernels]
    layer.apply_accumulated_gradients(learning_rate=0.1, batch_size=2)
    for kernel, before in zip(layer.kernels, weights_after_first_apply):
        assert kernel.weights == pytest.approx(before)


def test_snapshot_state_and_restore_state_round_trip():

    layer = _layer_with_state([0.0] * 9, height=3, width=3, kernel_size=2, channel_count=2)
    layer.kernels[0].weights = [1.0, 2.0, 3.0, 4.0]
    layer.kernels[0].bias = 0.5
    layer.kernels[1].weights = [-1.0, -2.0, -3.0, -4.0]
    layer.kernels[1].bias = -0.5

    snapshot = layer.snapshot_state()
    assert snapshot == [([1.0, 2.0, 3.0, 4.0], 0.5), ([-1.0, -2.0, -3.0, -4.0], -0.5)]
    assert len(snapshot) == layer.channel_count  # once per kernel, not once per unit

    for kernel in layer.kernels:
        kernel.weights = [9.0] * 4
        kernel.bias = 9.0

    layer.restore_state(snapshot)

    for kernel, (weights, bias) in zip(layer.kernels, snapshot):
        assert kernel.weights == weights
        assert kernel.bias == bias


def test_randomize_fan_in_aware_randomizes_every_kernel():

    random.seed(0)
    layer = _layer_with_state([0.0] * 9, height=3, width=3, kernel_size=2, channel_count=3)
    layer.randomize_fan_in_aware()

    for kernel in layer.kernels:
        assert len(set(kernel.weights)) > 1  # not all zero/identical

    all_weights = [w for kernel in layer.kernels for w in kernel.weights]
    assert len(set(all_weights)) > 1  # kernels drew independently, not identically


def test_gradient_check_against_a_numerically_perturbed_loss():

    # loss L = sum of every unit's post-ReLU activation, so dL/dz_i = 1 if active else 0; with
    # that as each unit's delta, every weight and bias gradient must match finite differences
    random.seed(0)
    height = width = 4
    input_layer = StateLayer(height * width, [(-10.0, 10.0)] * (height * width))
    input_layer.update_state(tuple(random.uniform(-2.0, 2.0) for _ in range(height * width)))
    layer = ConvLayer(input_layer=input_layer, input_height=height, input_width=width, kernel_size=2, channel_count=2)
    layer.randomize_fan_in_aware()

    def total_loss() -> float:
        layer.forward()
        return sum(unit.value() for unit in layer.nodes)

    total_loss()
    for unit in layer.nodes:
        unit.delta = 1.0 if unit.value() > 0.0 else 0.0
    layer.accumulate_gradients()

    epsilon = 1e-5
    for kernel in layer.kernels:
        for i in range(len(kernel.weights)):
            original = kernel.weights[i]
            kernel.weights[i] = original + epsilon
            loss_plus = total_loss()
            kernel.weights[i] = original - epsilon
            loss_minus = total_loss()
            kernel.weights[i] = original

            numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
            assert kernel._weight_gradient_accum[i] == pytest.approx(numerical_gradient, abs=1e-4)

        original_bias = kernel.bias
        kernel.bias = original_bias + epsilon
        loss_plus = total_loss()
        kernel.bias = original_bias - epsilon
        loss_minus = total_loss()
        kernel.bias = original_bias

        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
        assert kernel._bias_gradient_accum == pytest.approx(numerical_gradient, abs=1e-4)


def _stacked_conv_layers(height: int, width: int, stride: int) -> tuple[StateLayer, ConvLayer, ConvLayer]:
    input_layer = StateLayer(height * width, [(-10.0, 10.0)] * (height * width))
    input_layer.update_state(tuple(random.uniform(-2.0, 2.0) for _ in range(height * width)))
    first = ConvLayer(input_layer=input_layer, input_height=height, input_width=width, kernel_size=2, channel_count=2)
    second = ConvLayer(
        input_layer=first,
        input_height=first.out_height,
        input_width=first.out_width,
        kernel_size=2,
        channel_count=3,
        stride=stride,
        input_channels=first.channel_count,
    )
    first.randomize_fan_in_aware()
    second.randomize_fan_in_aware()
    return input_layer, first, second


@pytest.mark.parametrize("stride", [1, 2])
def test_downstream_sum_matches_a_brute_force_scan_over_every_unit(stride):

    # the reverse map is an optimization: it must equal a scan summing delta * weight over every
    # receptive-field slot that reads the input node
    random.seed(1)
    _input_layer, first, second = _stacked_conv_layers(height=7, width=7, stride=stride)
    first.forward()
    second.forward()
    for unit in second.nodes:
        unit.delta = random.uniform(-1.0, 1.0)

    for own_index, upstream in enumerate(first.nodes):
        brute_force = sum(
            unit.delta * unit.kernel.weights[weight_index]
            for unit in second.nodes
            for weight_index, node in enumerate(unit.input_nodes)
            if node is upstream
        )
        assert second.downstream_sum(own_index) == pytest.approx(brute_force, abs=1e-12)


def test_downstream_sum_is_zero_for_an_input_no_receptive_field_reads():

    # stride 3 with kernel_size 2 over a 5x5 input reads rows/cols {0, 1, 3, 4} only - row/col 2
    # is never in any receptive field, so its reverse-map entry is empty
    input_layer = StateLayer(25, [(-10.0, 10.0)] * 25)
    layer = ConvLayer(input_layer=input_layer, input_height=5, input_width=5, kernel_size=2, channel_count=1, stride=3)
    for unit in layer.nodes:
        unit.delta = 1.0

    skipped = 2 * 5 + 2  # row 2, col 2
    assert layer._fan_out[skipped] == []
    assert layer.downstream_sum(skipped) == 0.0


@pytest.mark.parametrize("stride", [1, 2])
def test_gradient_check_through_two_stacked_conv_layers(stride):

    # loss L = sum of the second conv layer's activations, so first-layer gradients flow back
    # through compute_hidden_deltas/downstream_sum
    random.seed(2)
    _input_layer, first, second = _stacked_conv_layers(height=7, width=7, stride=stride)

    def total_loss() -> float:
        first.forward()
        second.forward()
        return sum(unit.value() for unit in second.nodes)

    total_loss()
    for unit in second.nodes:
        unit.delta = 1.0 if unit.value() > 0.0 else 0.0
    first.compute_hidden_deltas(second)
    first.accumulate_gradients()
    second.accumulate_gradients()

    # guard against a vacuous pass: some first-layer units must actually carry gradient
    assert any(unit.delta != 0.0 for unit in first.nodes)

    epsilon = 1e-5
    for layer in (first, second):
        for kernel in layer.kernels:
            parameters = [(kernel.weights, i, kernel._weight_gradient_accum[i]) for i in range(len(kernel.weights))]
            for weights, i, analytic in parameters:
                original = weights[i]
                weights[i] = original + epsilon
                loss_plus = total_loss()
                weights[i] = original - epsilon
                loss_minus = total_loss()
                weights[i] = original
                assert analytic == pytest.approx((loss_plus - loss_minus) / (2 * epsilon), abs=1e-4)

            original_bias = kernel.bias
            kernel.bias = original_bias + epsilon
            loss_plus = total_loss()
            kernel.bias = original_bias - epsilon
            loss_minus = total_loss()
            kernel.bias = original_bias
            assert kernel._bias_gradient_accum == pytest.approx((loss_plus - loss_minus) / (2 * epsilon), abs=1e-4)
