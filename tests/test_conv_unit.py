import pytest

from indrajala_ml.model.conv_kernel import ConvKernel
from indrajala_ml.model.conv_unit import ConvUnit
from indrajala_ml.model.state_node import StateNode
from tests.helpers import approx


def _conv_unit(weights: list[float], bias: float, values: list[float]) -> ConvUnit:
    # kernel_size=1, in_channels=len(weights) passes ConvKernel's size check for any length
    kernel = ConvKernel(kernel_size=1, in_channels=len(weights), weights=list(weights), bias=bias)
    input_nodes = [StateNode(v) for v in values]
    unit = ConvUnit(input_nodes=input_nodes, kernel=kernel)
    unit.forward()
    return unit


def test_forward_relu_positive_branch_passes_z_through():

    # z = 0.5*2.0 + (-0.5)*(-3.0) + 0.1 = 1.0 + 1.5 + 0.1 = 2.6 (positive) -> relu(2.6) = 2.6
    unit = _conv_unit([0.5, -0.5], 0.1, [2.0, -3.0])

    assert unit.value() == approx(2.6)


def test_forward_relu_negative_branch_is_zero():

    # z = 0.1*1.0 - 5.0 = -4.9 (negative) -> relu(-4.9) = 0.0
    unit = _conv_unit([0.1], -5.0, [1.0])

    assert unit.value() == 0.0


def test_constructor_rejects_a_receptive_field_size_mismatch():

    kernel = ConvKernel(kernel_size=1, in_channels=2, weights=[0.1, 0.2])
    with pytest.raises(AssertionError):
        ConvUnit(input_nodes=[StateNode(1.0)], kernel=kernel)  # only 1 input, kernel needs 2


def test_compute_hidden_delta_propagates_downstream_when_active():

    unit = _conv_unit([0.5, -0.5], 0.1, [2.0, -3.0])  # active (value=2.6 > 0)

    unit.compute_hidden_delta(downstream_sum=-0.5 * 0.8)

    assert unit.delta == approx(-0.5 * 0.8)


def test_compute_hidden_delta_is_zero_when_the_unit_is_dead():

    unit = _conv_unit([0.1], -5.0, [1.0])  # dead (value=0.0)

    unit.compute_hidden_delta(downstream_sum=-0.5 * 0.8)

    assert unit.delta == 0.0


def test_compute_output_delta_raises_since_conv_unit_is_hidden_layer_only():

    unit = _conv_unit([0.5, -0.5], 0.1, [2.0, -3.0])

    with pytest.raises(NotImplementedError):
        unit.compute_output_delta(1.0)


def test_accumulate_gradient_delegates_into_the_shared_kernel():

    unit = _conv_unit([0.5, -0.5], 0.1, [2.0, -3.0])
    unit.delta = 0.3

    unit.accumulate_gradient()

    # accum_w = delta * receptive_field_value, per weight; accum_b = delta
    assert unit.kernel._weight_gradient_accum == approx([0.3 * 2.0, 0.3 * -3.0])
    assert unit.kernel._bias_gradient_accum == approx(0.3)


def test_accumulate_gradient_from_two_units_sharing_one_kernel_sums_into_it():

    kernel = ConvKernel(kernel_size=1, in_channels=1, weights=[0.5], bias=0.1)
    unit_a = ConvUnit(input_nodes=[StateNode(2.0)], kernel=kernel)
    unit_b = ConvUnit(input_nodes=[StateNode(3.0)], kernel=kernel)
    unit_a.forward()
    unit_b.forward()
    unit_a.delta = 0.2
    unit_b.delta = 0.4

    unit_a.accumulate_gradient()
    unit_b.accumulate_gradient()

    assert kernel._weight_gradient_accum == approx([0.2 * 2.0 + 0.4 * 3.0])
    assert kernel._bias_gradient_accum == approx(0.2 + 0.4)
