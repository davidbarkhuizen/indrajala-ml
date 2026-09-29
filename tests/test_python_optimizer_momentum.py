from collections.abc import Callable

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.state_layer import StateLayer
from indrajala_ml.model.state_node import StateNode
from indrajala_ml.model.update_rules import SGD, Momentum, UpdateRule
from tests.helpers import LayerOptimizer, WeightSets, approx


def _node(rule: UpdateRule, weight: float, bias: float) -> tuple[BackpropNode, Callable[[float], None]]:
    # a one-input node reading x=1.0, and its single-example step under rule
    node = BackpropNode(input_nodes=[StateNode(1.0)])
    node.update_input_weights([weight])
    node.bias = bias
    return node, LayerOptimizer(WeightSets(node), rule).apply_single(node)


def test_first_step_matches_plain_sgd_since_there_is_no_prior_delta():

    # with no previous step, the momentum term is momentum * 0.0
    momentum_node, momentum_step = _node(Momentum(0.9), weight=0.5, bias=0.1)
    plain_node, plain_step = _node(SGD(), weight=0.5, bias=0.1)

    momentum_node.delta = 0.2
    plain_node.delta = 0.2
    momentum_step(0.1)
    plain_step(0.1)

    assert momentum_node.input_node_weights[0] == approx(plain_node.input_node_weights[0])
    assert momentum_node.bias == approx(plain_node.bias)


def test_second_step_adds_the_momentum_term_by_hand():

    # hand-derived, eq. (9) u = m * u + g / B; w - lr * u, with x=1.0, delta=0.2 both steps,
    # learning_rate=0.1, momentum=0.9:
    #   step 1: u = 0.9*0.0 + 0.2*1.0 = 0.2 -> weight = 0.5 - 0.1*0.2 = 0.48
    #           u_b = 0.9*0.0 + 0.2 = 0.2 -> bias = 0.1 - 0.1*0.2 = 0.08
    #   step 2: u = 0.9*0.2 + 0.2 = 0.38 -> weight = 0.48 - 0.1*0.38 = 0.442
    #           u_b = 0.9*0.2 + 0.2 = 0.38 -> bias = 0.08 - 0.1*0.38 = 0.042
    node, step = _node(Momentum(0.9), weight=0.5, bias=0.1)

    node.delta = 0.2
    step(0.1)
    assert node.input_node_weights[0] == approx(0.48)
    assert node.bias == approx(0.08)

    node.delta = 0.2
    step(0.1)
    assert node.input_node_weights[0] == approx(0.442)
    assert node.bias == approx(0.042)


def test_a_rate_change_scales_the_whole_velocity_by_hand():

    # where eq. (9) and Rumelhart et al.'s eq. (10) differ: the rate doubles on step 2, and eq. (9)
    # applies the new rate to the whole velocity, past gradients included. As above, then
    # learning_rate=0.2 on step 2:
    #   step 2: u = 0.9*0.2 + 0.2 = 0.38 -> weight = 0.48 - 0.2*0.38 = 0.404, bias = 0.08 - 0.076 = 0.004
    # eq. (10) would give v = 0.2*0.2 + 0.9*0.02 = 0.058 -> weight 0.422, bias 0.022
    node, step = _node(Momentum(0.9), weight=0.5, bias=0.1)

    node.delta = 0.2
    step(0.1)
    node.delta = 0.2
    step(0.2)

    assert node.input_node_weights[0] == approx(0.404)
    assert node.bias == approx(0.004)


def test_zero_momentum_is_bit_identical_to_plain_sgd_across_many_steps():

    momentum_node, momentum_step = _node(Momentum(0.0), weight=0.5, bias=0.1)
    plain_node, plain_step = _node(SGD(), weight=0.5, bias=0.1)

    for delta in [0.2, -0.1, 0.05, 0.3, -0.4]:
        momentum_node.delta = delta
        plain_node.delta = delta
        momentum_step(0.1)
        plain_step(0.1)

    # at momentum 0.0, eq. (9) is exactly SGD's w - lr * (g / B)
    assert momentum_node.input_node_weights[0] == plain_node.input_node_weights[0]
    assert momentum_node.bias == plain_node.bias


def test_each_node_of_a_layer_keeps_its_own_velocity():

    # the state is per node (keyed by layer index, then node index), not shared across the layer:
    # two nodes with different deltas step by their own velocities
    input_layer = StateLayer(1, [(-10.0, 10.0)])
    input_layer.update_state((1.0,))
    layer = BackpropLayer(size=2, input_layer=input_layer)
    for node in layer.nodes:
        node.update_input_weights([0.5])
        node.bias = 0.1
    optimizer = LayerOptimizer(layer, Momentum(0.9))

    for _ in range(2):
        layer.nodes[0].delta = 0.2
        layer.nodes[1].delta = -0.2
        layer.accumulate_gradients()
        optimizer.apply(0.1, batch_size=1)

    # as test_second_step_adds_the_momentum_term_by_hand, and its mirror image
    assert layer.nodes[0].input_node_weights[0] == approx(0.442)
    assert layer.nodes[1].input_node_weights[0] == approx(0.558)
    assert optimizer.state[0] == ([approx([0.38])], [approx(0.38)])
    assert optimizer.state[1] == ([approx([-0.38])], [approx(-0.38)])
