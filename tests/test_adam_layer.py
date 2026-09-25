import pytest

from indrajala_ml.model.adam_layer import make_adam_node_cls
from indrajala_ml.model.state_node import StateNode

BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8


def _step_count(node: object) -> int:
    # AdamBackpropNode's t, private to make_adam_node_cls's class
    return vars(node)["_t"]


def _adam_node(weight: float, bias: float):
    node_cls = make_adam_node_cls(BETA1, BETA2, EPSILON)
    x = StateNode(1.0)
    node = node_cls(input_nodes=[x])
    node.update_input_weights([weight])
    node.bias = bias
    return node


def test_two_apply_gradient_steps_match_hand_derived_moment_accumulation():

    # hand-derived, two steps so m and v accumulate: x=1.0, delta=0.2 both steps,
    # learning_rate=0.1, m=v=0 at the start:
    #   step 1: g=0.2, m=0.1*0.2=0.02, v=0.001*0.04=0.00004, bias_correction1=0.1,
    #           bias_correction2=0.001, m_hat=0.2, v_hat=0.04, sqrt(v_hat)=0.2 ->
    #           update = 0.1*0.2/(0.2+1e-8) ~= 0.1 -> w=0.4000000049999997, b=4.999999733690252e-09
    #   step 2: m=0.9*0.02+0.1*0.2=0.038, v=0.999*0.00004+0.001*0.04=0.00007996,
    #           bias_correction1=0.19, bias_correction2=0.001999, m_hat=0.2,
    #           v_hat=0.04000000200100050... -> update ~= 0.1 again ->
    #           w=0.3000000100000002, b=-0.0999999899999998
    node = _adam_node(weight=0.5, bias=0.1)

    node.delta = 0.2
    node.apply_gradient(0.1)
    assert node.input_node_weights[0] == pytest.approx(0.4000000049999997)
    assert node.bias == pytest.approx(4.999999733690252e-09, abs=1e-9)

    node.delta = 0.2
    node.apply_gradient(0.1)
    assert node.input_node_weights[0] == pytest.approx(0.3000000100000002)
    assert node.bias == pytest.approx(-0.0999999899999998)


def test_step_count_increments_once_per_apply_call():

    # t, the per-node step count behind bias correction
    node = _adam_node(weight=0.5, bias=0.1)
    assert _step_count(node) == 0

    node.delta = 0.2
    node.apply_gradient(0.1)
    assert _step_count(node) == 1

    node.delta = 0.2
    node.apply_gradient(0.1)
    assert _step_count(node) == 2
