from indrajala_ml.model.adam_backprop_classifier_network import AdamBackpropClassifierNetwork
from indrajala_ml.model.layers.python.backprop_node import BackpropNode
from indrajala_ml.model.layers.python.state_node import StateNode
from indrajala_ml.model.specs.update_rules import Adam
from tests.helpers import LayerOptimizer, WeightSets, approx

BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8


def _adam_node(weight: float, bias: float) -> tuple[BackpropNode, LayerOptimizer]:
    # a one-input node reading x=1.0, and the optimizer stepping it under Adam
    node = BackpropNode(input_nodes=[StateNode(1.0)])
    node.update_input_weights([weight])
    node.bias = bias
    return node, LayerOptimizer(WeightSets(node), Adam(BETA1, BETA2, EPSILON))


def test_two_steps_match_hand_derived_moment_accumulation():

    # hand-derived, two steps so m and v accumulate: x=1.0, delta=0.2 both steps,
    # learning_rate=0.1, m=v=0 at the start:
    #   step 1: g=0.2, m=0.1*0.2=0.02, v=0.001*0.04=0.00004, bias_correction1=0.1,
    #           bias_correction2=0.001, m_hat=0.2, v_hat=0.04, sqrt(v_hat)=0.2 ->
    #           update = 0.1*0.2/(0.2+1e-8) ~= 0.1 -> w=0.4000000049999997, b=4.999999733690252e-09
    #   step 2: m=0.9*0.02+0.1*0.2=0.038, v=0.999*0.00004+0.001*0.04=0.00007996,
    #           bias_correction1=0.19, bias_correction2=0.001999, m_hat=0.2,
    #           v_hat=0.04000000200100050... -> update ~= 0.1 again ->
    #           w=0.3000000100000002, b=-0.0999999899999998
    node, optimizer = _adam_node(weight=0.5, bias=0.1)
    step = optimizer.apply_single(node)

    node.delta = 0.2
    step(0.1)
    assert node.input_node_weights[0] == approx(0.4000000049999997)
    assert node.bias == approx(4.999999733690252e-09, abs=1e-9)

    node.delta = 0.2
    step(0.1)
    assert node.input_node_weights[0] == approx(0.3000000100000002)
    assert node.bias == approx(-0.0999999899999998)


def test_step_count_increments_once_per_step():

    # t, the step count behind bias correction
    node, optimizer = _adam_node(weight=0.5, bias=0.1)
    step = optimizer.apply_single(node)
    assert optimizer.optimizer.t == 0

    node.delta = 0.2
    step(0.1)
    assert optimizer.optimizer.t == 1

    node.delta = 0.2
    step(0.1)
    assert optimizer.optimizer.t == 2


def test_the_network_counts_one_step_per_learn_call_not_one_per_layer():

    # one t for the whole network, as each node's own count was: every trainable node stepped
    # exactly once per learn()/learn_batch()
    network = AdamBackpropClassifierNetwork.randomized([3, 2], 2, [(-1.0, 1.0)] * 2)

    network.learn(0.1, (0.5, -0.5), 1.0)
    assert network.optimizer.t == 1

    network.learn_batch(0.1, [((0.5, -0.5), 1.0), ((-0.5, 0.5), 0.0)])
    assert network.optimizer.t == 2
