import pytest

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.state_layer import StateLayer


def _layer(size: int, dimension: int, state: tuple[float, ...]) -> BackpropLayer:
    input_layer = StateLayer(dimension, square_bounds(10.0, dimension))
    input_layer.update_state(state)
    return BackpropLayer(size=size, input_layer=input_layer)


def test_apply_gradients_matches_calling_apply_gradient_on_every_node_by_hand():

    layer = _layer(3, 2, (2.0, -3.0))
    for i, node in enumerate(layer.nodes):
        node.update_input_weights([0.5, -0.5])
        node.bias = 0.1
        node.delta = 0.2 + i * 0.1

    expected = [
        (
            [w - 0.1 * node.delta * x for w, x in zip(node.input_node_weights, [2.0, -3.0])],
            node.bias - 0.1 * node.delta,
        )
        for node in layer.nodes
    ]

    layer.apply_gradients(0.1)

    for node, (expected_weights, expected_bias) in zip(layer.nodes, expected):
        assert node.input_node_weights == pytest.approx(expected_weights)
        assert node.bias == pytest.approx(expected_bias)


def test_accumulate_then_apply_accumulated_gradients_matches_per_node_split():

    layer = _layer(2, 1, (2.0,))
    for node in layer.nodes:
        node.update_input_weights([0.5])
        node.bias = 0.1

    for delta in [0.2, -0.1]:
        for node in layer.nodes:
            node.delta = delta
        layer.accumulate_gradients()

    layer.apply_accumulated_gradients(0.1, batch_size=2)

    # accum for weight = 0.2*2.0 + (-0.1)*2.0 = 0.2 -> weight -= 0.1 * (0.2/2)
    # accum for bias = 0.2 + (-0.1) = 0.1 -> bias -= 0.1 * (0.1/2)
    for node in layer.nodes:
        assert node.input_node_weights[0] == pytest.approx(0.5 - 0.1 * (0.2 / 2))
        assert node.bias == pytest.approx(0.1 - 0.1 * (0.1 / 2))



def test_downstream_sum_is_the_dense_delta_times_weight_sum():

    layer = _layer(2, 3, (1.0, 2.0, 3.0))
    layer.nodes[0].update_input_weights([0.5, -0.5, 0.25])
    layer.nodes[1].update_input_weights([-1.0, 2.0, 0.75])
    layer.nodes[0].delta = 0.2
    layer.nodes[1].delta = -0.4

    assert layer.downstream_sum(0) == pytest.approx(0.2 * 0.5 + -0.4 * -1.0)
    assert layer.downstream_sum(1) == pytest.approx(0.2 * -0.5 + -0.4 * 2.0)
    assert layer.downstream_sum(2) == pytest.approx(0.2 * 0.25 + -0.4 * 0.75)


def test_compute_hidden_deltas_matches_calling_compute_hidden_delta_on_every_node_by_hand():

    hidden = _layer(3, 2, (2.0, -3.0))
    for node in hidden.nodes:
        node.update_input_weights([0.3, -0.2])
        node.forward()
    next_layer = BackpropLayer(size=2, input_layer=hidden)
    for i, node in enumerate(next_layer.nodes):
        node.update_input_weights([0.5, -0.25, 1.0 + i])
        node.delta = 0.1 * (i + 1)

    for own_index, node in enumerate(hidden.nodes):
        node.compute_hidden_delta(next_layer.nodes, own_index)
    expected = [node.delta for node in hidden.nodes]
    for node in hidden.nodes:
        node.delta = 0.0

    hidden.compute_hidden_deltas(next_layer)

    assert [node.delta for node in hidden.nodes] == expected

def test_snapshot_state_and_restore_state_round_trip():

    layer = _layer(2, 2, (0.0, 0.0))
    for i, node in enumerate(layer.nodes):
        node.update_input_weights([0.5 + i, -0.5 - i])
        node.bias = 0.1 * i

    snapshot = layer.snapshot_state()
    assert snapshot == [([0.5, -0.5], 0.0), ([1.5, -1.5], 0.1)]

    for node in layer.nodes:
        node.update_input_weights([9.0, 9.0])
        node.bias = 9.0

    layer.restore_state(snapshot)

    for node, (weights, bias) in zip(layer.nodes, snapshot):
        assert node.input_node_weights == weights
        assert node.bias == bias


class _CallCountingLayer:
    """A fake trainable layer - exercises BackpropNetworkBase's own gradient/persistence
    methods to confirm they call each of these once per *layer*, not once per node, which is
    exactly the seam a convolutional layer (many nodes sharing one kernel) depends on."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.accumulate_calls = 0
        self.apply_accumulated_calls = 0
        self.apply_gradients_calls = 0
        self.snapshot_calls = 0
        self.restore_calls = 0
        self.last_apply_args = None
        self.last_restored = None

    def accumulate_gradients(self):
        self.accumulate_calls += 1

    def apply_accumulated_gradients(self, learning_rate, batch_size):
        self.apply_accumulated_calls += 1
        self.last_apply_args = (learning_rate, batch_size)

    def apply_gradients(self, learning_rate):
        self.apply_gradients_calls += 1

    def snapshot_state(self):
        self.snapshot_calls += 1
        return f"fake-snapshot-{id(self)}"

    def restore_state(self, layer_snapshot):
        self.restore_calls += 1
        self.last_restored = layer_snapshot


def test_network_level_methods_dispatch_once_per_layer_not_once_per_node():

    network = BackpropClassifierNetwork.randomized([4], 2, square_bounds(10.0))

    fake_hidden = _CallCountingLayer(nodes=[object(), object(), object(), object()])
    fake_output = _CallCountingLayer(nodes=[object()])
    network.trainable_layers = [fake_hidden, fake_output]

    network._accumulate_gradients()
    assert fake_hidden.accumulate_calls == 1
    assert fake_output.accumulate_calls == 1

    network._apply_accumulated_gradients(0.1, batch_size=3)
    assert fake_hidden.apply_accumulated_calls == 1
    assert fake_hidden.last_apply_args == (0.1, 3)
    assert fake_output.apply_accumulated_calls == 1

    network._apply_gradients(0.1)
    assert fake_hidden.apply_gradients_calls == 1
    assert fake_output.apply_gradients_calls == 1

    snapshot = network.snapshot()
    assert fake_hidden.snapshot_calls == 1
    assert fake_output.snapshot_calls == 1
    assert snapshot == [f"fake-snapshot-{id(fake_hidden)}", f"fake-snapshot-{id(fake_output)}"]

    network.restore(["restored-hidden", "restored-output"])
    assert fake_hidden.restore_calls == 1
    assert fake_hidden.last_restored == "restored-hidden"
    assert fake_output.restore_calls == 1
    assert fake_output.last_restored == "restored-output"
