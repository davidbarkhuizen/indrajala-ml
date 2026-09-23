import random

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

    # 3x3 input, row-major: [[1,2,3],[4,5,6],[7,8,9]]; kernel_size=2, stride=1 -> 2x2 output.
    # kernel weights [1,0,0,0] (order: (kr=0,kc=0),(0,1),(1,0),(1,1)) picks each window's
    # top-left element - which is just the input value at (row,col) itself for a stride-1,
    # top-left-anchored window - independently verified by hand below, not re-derived from the
    # implementation under test:
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

    # 4x4 input, all zero except one pixel at (row=1, col=2) set to 1.0; kernel_size=2,
    # stride=1, all-ones kernel, zero bias - only output positions whose 2x2 window covers
    # (1, 2) should be nonzero, and each should be exactly 1.0 (only one input contributes).
    # Windows covering (1,2): top-left corner at (0,1),(0,2),(1,1),(1,2) - i.e. output positions
    # (0,1),(0,2),(1,1),(1,2) in a 3x3 output grid (out_height=out_width=3 for 4x4 input,
    # kernel_size=2).
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


def test_apply_gradients_matches_a_hand_computed_single_example():

    layer = _layer_with_state([1, 2, 3, 4, 5, 6, 7, 8, 9], height=3, width=3, kernel_size=2)
    layer.kernels[0].weights = [1.0, 0.0, 0.0, 0.0]
    layer.kernels[0].bias = 0.0
    layer.forward()

    for unit in layer.nodes:
        unit.delta = 0.1  # a fixed downstream delta at every position, for a simple hand check

    # accum per weight index = sum over positions of delta * that position's own receptive
    # field value at that index. Position windows (top-left-anchored values): [1,2,4,5],
    # [2,3,5,6], [4,5,7,8], [5,6,8,9] - index 0 (top-left) across all 4 positions: 1+2+4+5=12
    layer.apply_gradients(learning_rate=0.1)

    expected_weight_0 = 1.0 - 0.1 * (0.1 * 1 + 0.1 * 2 + 0.1 * 4 + 0.1 * 5)
    assert layer.kernels[0].weights[0] == pytest.approx(expected_weight_0)
    expected_bias = 0.0 - 0.1 * (0.1 * 4)  # 4 positions, delta=0.1 each, summed then batch_size=1
    assert layer.kernels[0].bias == pytest.approx(expected_bias)


def test_apply_accumulated_gradients_applies_once_per_kernel_not_once_per_unit():

    # channel_count=2, several units per channel - apply_accumulated_gradients must move each
    # kernel's weights by exactly the summed-then-batch-divided amount once, not repeatedly
    # (which apply_accumulated_gradient's own accumulator reset already guards against, but
    # this test exercises it through the real layer/kernel wiring, not in isolation)
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

    # the standard, rigorous validation for a backward-pass formula: loss
    # L = sum of every unit's (post-ReLU) activation, so dL/da_i = 1 and (via the ReLU
    # derivative) dL/dz_i = 1 if active else 0 - setting exactly that as each unit's delta
    # before accumulate_gradients() is the real gradient this loss produces, checked here
    # against a numerically-perturbed finite-difference estimate for every weight and bias.
    random.seed(0)
    height = width = 4
    input_layer = StateLayer(height * width, [(-10.0, 10.0)] * (height * width))
    input_layer.update_state(tuple(random.uniform(-2.0, 2.0) for _ in range(height * width)))
    layer = ConvLayer(
        input_layer=input_layer, input_height=height, input_width=width, kernel_size=2, channel_count=2
    )
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
