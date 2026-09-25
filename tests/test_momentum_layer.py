import pytest

from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.momentum_layer import make_momentum_layer_cls, make_momentum_node_cls
from indrajala_ml.model.state_layer import StateLayer
from indrajala_ml.model.state_node import StateNode


def _momentum_node(momentum: float, weight: float, bias: float):
    node_cls = make_momentum_node_cls(momentum)
    x = StateNode(1.0)
    node = node_cls(input_nodes=[x])
    node.update_input_weights([weight])
    node.bias = bias
    return node


def test_first_apply_gradient_matches_plain_sgd_since_there_is_no_prior_delta():

    # with no previous step, the momentum term is momentum * 0.0
    momentum_node = _momentum_node(0.9, weight=0.5, bias=0.1)
    plain_node = BackpropNode(input_nodes=[StateNode(1.0)])
    plain_node.update_input_weights([0.5])
    plain_node.bias = 0.1

    momentum_node.delta = 0.2
    plain_node.delta = 0.2
    momentum_node.apply_gradient(0.1)
    plain_node.apply_gradient(0.1)

    assert momentum_node.input_node_weights[0] == pytest.approx(plain_node.input_node_weights[0])
    assert momentum_node.bias == pytest.approx(plain_node.bias)


def test_second_apply_gradient_adds_the_momentum_term_by_hand():

    # hand-derived, eq. (9) u = m * u + g / B; w - lr * u, with x=1.0, delta=0.2 both steps,
    # learning_rate=0.1, momentum=0.9:
    #   step 1: u = 0.9*0.0 + 0.2*1.0 = 0.2 -> weight = 0.5 - 0.1*0.2 = 0.48
    #           u_b = 0.9*0.0 + 0.2 = 0.2 -> bias = 0.1 - 0.1*0.2 = 0.08
    #   step 2: u = 0.9*0.2 + 0.2 = 0.38 -> weight = 0.48 - 0.1*0.38 = 0.442
    #           u_b = 0.9*0.2 + 0.2 = 0.38 -> bias = 0.08 - 0.1*0.38 = 0.042
    node = _momentum_node(0.9, weight=0.5, bias=0.1)

    node.delta = 0.2
    node.apply_gradient(0.1)
    assert node.input_node_weights[0] == pytest.approx(0.48)
    assert node.bias == pytest.approx(0.08)

    node.delta = 0.2
    node.apply_gradient(0.1)
    assert node.input_node_weights[0] == pytest.approx(0.442)
    assert node.bias == pytest.approx(0.042)


def test_a_rate_change_scales_the_whole_velocity_by_hand():

    # where eq. (9) and Rumelhart et al.'s eq. (10) differ: the rate doubles on step 2, and eq. (9)
    # applies the new rate to the whole velocity, past gradients included. As above, then
    # learning_rate=0.2 on step 2:
    #   step 2: u = 0.9*0.2 + 0.2 = 0.38 -> weight = 0.48 - 0.2*0.38 = 0.404, bias = 0.08 - 0.076 = 0.004
    # eq. (10) would give v = 0.2*0.2 + 0.9*0.02 = 0.058 -> weight 0.422, bias 0.022
    node = _momentum_node(0.9, weight=0.5, bias=0.1)

    node.delta = 0.2
    node.apply_gradient(0.1)
    node.delta = 0.2
    node.apply_gradient(0.2)

    assert node.input_node_weights[0] == pytest.approx(0.404)
    assert node.bias == pytest.approx(0.004)


def test_zero_momentum_is_bit_identical_to_plain_sgd_across_many_steps():

    momentum_node = _momentum_node(0.0, weight=0.5, bias=0.1)
    plain_node = BackpropNode(input_nodes=[StateNode(1.0)])
    plain_node.update_input_weights([0.5])
    plain_node.bias = 0.1

    for delta in [0.2, -0.1, 0.05, 0.3, -0.4]:
        momentum_node.delta = delta
        plain_node.delta = delta
        momentum_node.apply_gradient(0.1)
        plain_node.apply_gradient(0.1)

    # at momentum 0.0, eq. (9) is exactly SGD's w - lr * (g / B)
    assert momentum_node.input_node_weights[0] == plain_node.input_node_weights[0]
    assert momentum_node.bias == plain_node.bias


def test_make_momentum_layer_cls_builds_nodes_of_the_configured_momentum_node_class():

    input_layer = StateLayer(2, [(-10.0, 10.0), (-10.0, 10.0)])
    layer_cls = make_momentum_layer_cls(0.5)
    layer = layer_cls(size=3, input_layer=input_layer)

    node_cls = make_momentum_node_cls(0.5)
    assert all(type(node).__name__ == node_cls.__name__ for node in layer.nodes)
    assert all(hasattr(node, "_weight_velocities") for node in layer.nodes)
