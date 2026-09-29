from indrajala_ml.model.conv_kernel import ConvKernel
from indrajala_ml.model.conv_layer import ConvLayer, ConvSpec
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.momentum_conv_multiclass_backprop_classifier_network import (
    MomentumConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import SGD, Momentum
from tests.helpers import LayerOptimizer, WeightSets, approx


def _accumulate_two_examples_of_two_positions(kernel: ConvKernel, delta: float) -> None:
    # per example, positions reading 1.0 and 2.0: accum w = 2 * (delta * 1.0 + delta * 2.0), b = 4 * delta
    for _example in range(2):
        kernel.accumulate_gradient(delta=delta, receptive_field_values=[1.0])
        kernel.accumulate_gradient(delta=delta, receptive_field_values=[2.0])


def test_two_steps_with_a_rate_change_by_hand():

    # hand-derived, eq. (9) u = m * u + g / B; w - lr * u, momentum=0.9, batch_size=2, delta=0.2 on
    # both steps, so g_w = 1.2 and g_b = 0.8 (positions summed, only the 2 examples averaged):
    #   step 1, lr=0.1: u_w = 0.6 -> weight = 0.5 - 0.06 = 0.44; u_b = 0.4 -> bias = 0.1 - 0.04 = 0.06
    #   step 2, lr=0.2: u_w = 0.9*0.6 + 0.6 = 1.14 -> weight = 0.44 - 0.228 = 0.212
    #                   u_b = 0.9*0.4 + 0.4 = 0.76 -> bias = 0.06 - 0.152 = -0.092
    # eq. (10) would give step 2's v_w = 0.2*0.6 + 0.9*0.06 = 0.174 -> weight 0.266
    kernel = ConvKernel(kernel_size=1, in_channels=1, weights=[0.5], bias=0.1)
    optimizer = LayerOptimizer(WeightSets(kernel), Momentum(0.9))

    _accumulate_two_examples_of_two_positions(kernel, delta=0.2)
    optimizer.apply(learning_rate=0.1, batch_size=2)
    assert kernel.weights == approx([0.44])
    assert kernel.bias == approx(0.06)

    _accumulate_two_examples_of_two_positions(kernel, delta=0.2)
    optimizer.apply(learning_rate=0.2, batch_size=2)
    assert kernel.weights == approx([0.212])
    assert kernel.bias == approx(-0.092)


def test_zero_momentum_is_bit_identical_to_the_plain_kernel_across_many_steps():

    momentum_kernel = ConvKernel(kernel_size=1, in_channels=2, weights=[0.5, -0.3], bias=0.1)
    plain_kernel = ConvKernel(kernel_size=1, in_channels=2, weights=[0.5, -0.3], bias=0.1)
    optimizers = {
        id(momentum_kernel): LayerOptimizer(WeightSets(momentum_kernel), Momentum(0.0)),
        id(plain_kernel): LayerOptimizer(WeightSets(plain_kernel), SGD()),
    }

    # batch size 3, not a power of two, so g / B rounds
    for delta in [0.2, -0.1, 0.05, 0.3, -0.4]:
        for kernel in (momentum_kernel, plain_kernel):
            for values in ([1.0, 0.3], [0.7, 0.2], [0.1, 0.9]):
                kernel.accumulate_gradient(delta, values)
            optimizers[id(kernel)].apply(learning_rate=0.1, batch_size=3)

    # at momentum 0.0, eq. (9) is exactly SGD's w - lr * (g / B)
    assert momentum_kernel.weights == plain_kernel.weights
    assert momentum_kernel.bias == plain_kernel.bias


def test_a_step_resets_the_accumulator_but_keeps_the_velocity():

    # an apply with nothing accumulated still steps by the decayed velocity: u = 0.9 * 0.2 = 0.18
    kernel = ConvKernel(kernel_size=1, in_channels=1, weights=[0.5], bias=0.1)
    optimizer = LayerOptimizer(WeightSets(kernel), Momentum(0.9))

    kernel.accumulate_gradient(delta=0.2, receptive_field_values=[1.0])
    optimizer.apply(0.1, batch_size=1)
    optimizer.apply(0.1, batch_size=1)

    assert kernel.weights == approx([0.5 - 0.1 * 0.2 - 0.1 * 0.18])
    assert kernel.bias == approx(0.1 - 0.1 * 0.2 - 0.1 * 0.18)


def test_the_momentum_conv_network_steps_plain_layers_by_its_optimizer():

    # the conv and dense layers carry no rule; the network's optimizer applies Momentum to every
    # kernel and node, and keeps a velocity per kernel once the conv layer has stepped
    network = MomentumConvMultiClassBackpropClassifierNetwork.randomized(
        4, 4, [ConvSpec(kernel_size=2, channel_count=3), PoolSpec(pool_size=3)], [2], 2, momentum=0.5
    )
    conv = network.conv_layers[0]
    assert type(conv) is ConvLayer
    assert all(type(kernel) is ConvKernel for kernel in conv.kernels)
    assert network.optimizer.rule == Momentum(0.5)

    network.learn(0.1, tuple([0.5] * 16), 1)

    # one velocity state per kernel: 3 channels of 2 x 2 x 1 weights
    assert len(network.optimizer._state[0]) == 3
    assert all(len(weights[0]) == 4 for weights, _bias in network.optimizer._state[0])
    # a pool layer has no weight sets, so no state
    assert network.optimizer._state[1] == []
