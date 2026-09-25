from indrajala_ml.model.conv_kernel import ConvKernel
from indrajala_ml.model.conv_layer import ConvLayer
from indrajala_ml.model.momentum_conv_layer import make_momentum_conv_layer_cls, make_momentum_kernel_cls
from indrajala_ml.model.state_layer import StateLayer
from tests.helpers import approx


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
    kernel = make_momentum_kernel_cls(0.9)(kernel_size=1, in_channels=1, weights=[0.5], bias=0.1)

    _accumulate_two_examples_of_two_positions(kernel, delta=0.2)
    kernel.apply_accumulated_gradient(learning_rate=0.1, batch_size=2)
    assert kernel.weights == approx([0.44])
    assert kernel.bias == approx(0.06)

    _accumulate_two_examples_of_two_positions(kernel, delta=0.2)
    kernel.apply_accumulated_gradient(learning_rate=0.2, batch_size=2)
    assert kernel.weights == approx([0.212])
    assert kernel.bias == approx(-0.092)


def test_zero_momentum_is_bit_identical_to_the_plain_kernel_across_many_steps():

    momentum_kernel = make_momentum_kernel_cls(0.0)(kernel_size=1, in_channels=2, weights=[0.5, -0.3], bias=0.1)
    plain_kernel = ConvKernel(kernel_size=1, in_channels=2, weights=[0.5, -0.3], bias=0.1)

    # batch size 3, not a power of two, so g / B rounds
    for delta in [0.2, -0.1, 0.05, 0.3, -0.4]:
        for kernel in (momentum_kernel, plain_kernel):
            for values in ([1.0, 0.3], [0.7, 0.2], [0.1, 0.9]):
                kernel.accumulate_gradient(delta, values)
            kernel.apply_accumulated_gradient(learning_rate=0.1, batch_size=3)

    # at momentum 0.0, eq. (9) is exactly SGD's w - lr * (g / B)
    assert momentum_kernel.weights == plain_kernel.weights
    assert momentum_kernel.bias == plain_kernel.bias


def test_apply_accumulated_gradient_resets_the_accumulator_but_keeps_the_velocity():

    # an apply with nothing accumulated still steps by the decayed velocity: u = 0.9 * 0.2 = 0.18
    kernel = make_momentum_kernel_cls(0.9)(kernel_size=1, in_channels=1, weights=[0.5], bias=0.1)

    kernel.accumulate_gradient(delta=0.2, receptive_field_values=[1.0])
    kernel.apply_accumulated_gradient(0.1, batch_size=1)
    kernel.apply_accumulated_gradient(0.1, batch_size=1)

    assert kernel.weights == approx([0.5 - 0.1 * 0.2 - 0.1 * 0.18])
    assert kernel.bias == approx(0.1 - 0.1 * 0.2 - 0.1 * 0.18)


def test_make_momentum_conv_layer_cls_builds_kernels_of_the_momentum_kernel_class():

    layer_cls = make_momentum_conv_layer_cls(0.5)
    layer = layer_cls(StateLayer(16, [(0.0, 1.0)] * 16), input_height=4, input_width=4, kernel_size=2, channel_count=3)

    assert issubclass(layer_cls, ConvLayer)
    assert all(type(kernel).__name__ == "MomentumConvKernel" for kernel in layer.kernels)
    assert all(vars(kernel)["_weight_velocities"] == [0.0] * 4 for kernel in layer.kernels)
