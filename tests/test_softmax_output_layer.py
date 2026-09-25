import pytest

from indrajala_ml.model.softmax_output_layer import SoftmaxOutputLayer
from indrajala_ml.model.state_layer import StateLayer


def _fixed_layer() -> tuple[StateLayer, SoftmaxOutputLayer]:
    # x=1.0, weights [1.0, 0.0, -1.0], biases 0: z = [1.0, 0.0, -1.0]
    input_layer = StateLayer(1, [(-10.0, 10.0)])
    input_layer.update_state((1.0,))

    layer = SoftmaxOutputLayer(size=3, input_layer=input_layer)
    for node, weight in zip(layer.nodes, [1.0, 0.0, -1.0]):
        node.update_input_weights([weight])

    return input_layer, layer


def test_forward_matches_a_hand_computed_softmax():

    # hand-derived with the textbook formula (not the implementation's max-subtracted form):
    # e^1 + e^0 + e^-1 = 4.086161269630487
    _, layer = _fixed_layer()

    layer.forward()

    activations = [node.value() for node in layer.nodes]
    assert activations == [
        pytest.approx(0.6652409557748219),
        pytest.approx(0.24472847105479767),
        pytest.approx(0.09003057317038046),
    ]
    assert sum(activations) == pytest.approx(1.0)


def test_compute_output_delta_matches_a_hand_computed_softmax_cross_entropy_delta():

    # softmax + cross-entropy: delta = activation - one_hot_target, target class 0
    _, layer = _fixed_layer()
    layer.forward()

    for node, target in zip(layer.nodes, [1.0, 0.0, 0.0]):
        node.compute_output_delta(target)

    deltas = [node.delta for node in layer.nodes]
    assert deltas == [
        pytest.approx(-0.3347590442251781),
        pytest.approx(0.24472847105479767),
        pytest.approx(0.09003057317038046),
    ]


def test_forward_is_invariant_to_a_constant_shift_in_every_z():

    # with max subtraction, shifting every z (here via the biases) changes nothing, bit for bit
    _, shifted_layer = _fixed_layer()
    for node in shifted_layer.nodes:
        node.bias = node.bias + 1000.0

    _, baseline_layer = _fixed_layer()

    shifted_layer.forward()
    baseline_layer.forward()

    shifted = [node.value() for node in shifted_layer.nodes]
    baseline = [node.value() for node in baseline_layer.nodes]
    assert shifted == baseline


def test_forward_does_not_overflow_for_a_very_large_z():

    # without max subtraction, exp(10_000) overflows
    _, layer = _fixed_layer()
    layer.nodes[0].bias = 10_000.0

    layer.forward()

    activations = [node.value() for node in layer.nodes]
    assert activations[0] == pytest.approx(1.0)
    assert activations[1] == pytest.approx(0.0)
    assert activations[2] == pytest.approx(0.0)
    assert sum(activations) == pytest.approx(1.0)


def test_node_forward_raises_since_activation_must_be_computed_jointly_by_the_layer():

    _, layer = _fixed_layer()

    with pytest.raises(NotImplementedError):
        layer.nodes[0].forward()


def test_a_softmax_layer_needs_at_least_two_nodes():

    input_layer = StateLayer(1, [(-10.0, 10.0)])
    input_layer.update_state((1.0,))

    with pytest.raises(AssertionError):
        SoftmaxOutputLayer(size=1, input_layer=input_layer)
