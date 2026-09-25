import pytest

from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.relu_layer import ReLULayer
from indrajala_ml.model.state_layer import StateLayer
from tests.helpers import approx


def _relu_node(weight: float, bias: float, x: float):
    # 1D input, 1 ReLU node - small enough to hand-compute z and the resulting activation exactly
    input_layer = StateLayer(1, [(-10.0, 10.0)])
    input_layer.update_state((x,))
    layer = ReLULayer(size=1, input_layer=input_layer)
    node = layer.nodes[0]
    node.update_input_weights([weight])
    node.bias = bias
    layer.forward()
    return node


def test_forward_relu_positive_branch_passes_z_through():

    # z = 0.5*2.0 + 0.1 = 1.1 (positive) -> relu(1.1) = 1.1
    node = _relu_node(0.5, 0.1, 2.0)

    assert node.value() == approx(1.1)


def test_forward_relu_negative_branch_is_zero():

    # z = 0.5*(-10.0) + 0.1 = -4.9 (negative) -> relu(-4.9) = 0.0
    node = _relu_node(0.5, 0.1, -10.0)

    assert node.value() == 0.0


def test_forward_relu_at_exactly_zero_is_zero():

    node = _relu_node(0.0, 0.0, 5.0)  # z = 0.0

    assert node.value() == 0.0


def test_compute_hidden_delta_propagates_downstream_when_active():

    # a=1.1 > 0 (active) - the ReLU derivative factor is 1.0, so delta is exactly the weighted
    # downstream sum, unmodified
    node = _relu_node(0.5, 0.1, 2.0)
    next_node = BackpropNode(input_nodes=[node])
    next_node.update_input_weights([0.8])
    next_node.delta = -0.5

    node.compute_hidden_delta([next_node], own_index=0)

    assert node.delta == approx(-0.5 * 0.8)


def test_compute_hidden_delta_is_zero_when_the_unit_is_dead():

    # a=0.0 (inactive, z<=0) - the ReLU derivative factor is 0.0, so delta is exactly zero
    # regardless of how large the downstream error is - the "dead unit" case
    node = _relu_node(0.5, 0.1, -10.0)
    next_node = BackpropNode(input_nodes=[node])
    next_node.update_input_weights([0.8])
    next_node.delta = -0.5

    node.compute_hidden_delta([next_node], own_index=0)

    assert node.delta == 0.0


def test_compute_output_delta_raises_since_relu_is_hidden_layer_only():

    node = _relu_node(0.5, 0.1, 2.0)

    with pytest.raises(NotImplementedError):
        node.compute_output_delta(1.0)
