"""
Dense batch-norm networks on each array backend: gradient checks under every rule, the running
averages in training and inference, the optimizer's per-parameter state, weight decay (D7), the
one-example refusal (D4), snapshot and checkpoint. The layers, and what one backend alone has, stay
in tests/test_batch_norm_array_network.py (numpy) and tests/test_batch_norm_rust_network.py (Rust,
with the parity tests); this module reuses their cases.
"""

import numpy as np
import pytest

from indrajala_ml.model.array_layer import sigmoid
from indrajala_ml.model.specs.update_rules import SGD, Adam, UpdateRule, WeightDecay
from tests.gradient_check import check_gradients
from tests.helpers import Backend, bits, to_numpy
from tests.test_batch_norm_array_network import EPSILON, NETWORKS, RULES, _network, _rows


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", NETWORKS)
@pytest.mark.parametrize("batch_size", [2, 5])
def test_every_gradient_matches_its_finite_difference(name: str, rule: UpdateRule, batch_size: int, backend: Backend):
    network = _network(name, rule, backend=backend)
    rows = _rows(batch_size, NETWORKS[name][1])
    # moved running averages and a trained step, so gamma and beta aren't at their initial values
    network.learn_batch(0.5, _rows(6, NETWORKS[name][1], seed=2))

    check_gradients(network, [state for state, _ in rows], [label for _, label in rows])


def test_the_running_averages_move_in_training_forward_passes_only(backend: Backend):
    network = _network(backend=backend)
    rows = _rows(6)
    prepared = network.prepare_dataset(rows)
    norm = network.layers[1]

    network.learn_batch(0.5, rows)
    trained = bits(norm.running_state())
    assert trained != bits((np.zeros(5), np.ones(5)))

    network.classify_rows(prepared)
    network.classify_state(rows[0][0])
    network.classify_row(prepared, 1)
    assert bits(norm.running_state()) == trained


def test_classifying_normalizes_with_the_running_averages(backend: Backend):
    network = _network(backend=backend)
    rows = _rows(6)
    network.learn_batch(0.5, rows)
    linear, norm, output = network.layers
    Z = np.array([state for state, _ in rows]) @ to_numpy(linear.W).T
    xhat = (Z - to_numpy(norm.running_mean)) / np.sqrt(to_numpy(norm.running_var) + EPSILON)
    hidden = sigmoid(to_numpy(norm.gamma) * xhat + to_numpy(norm.beta))
    expected = sigmoid(hidden @ to_numpy(output.W).T + to_numpy(output.b))

    assert network.classify_rows(network.prepare_dataset(rows)) == np.argmax(expected, axis=1).tolist()
    assert [network.classify_state(state) for state, _ in rows] == np.argmax(expected, axis=1).tolist()


def test_the_optimizers_state_is_per_parameter(backend: Backend):
    network = _network(rule=Adam(), backend=backend)
    network.learn_batch(0.1, _rows(6))
    layers = network.optimizer.state().layers

    assert [array.shape for array in layers[0]] == [(5, 4), (5, 4)]  # the linear layer's m and v
    assert [array.shape for array in layers[1]] == [(5,)] * 4  # gamma's m and v, beta's m and v
    assert [array.shape for array in layers[2]] == [(3, 5), (3, 5), (3,), (3,)]


def test_weight_decay_decays_the_linear_layers_w_and_neither_gamma_nor_beta(backend: Backend):
    rows = _rows(6)
    sgd, decayed = _network(rule=SGD(), backend=backend), _network(rule=WeightDecay(0.1), backend=backend)

    for network in (sgd, decayed):
        network.learn_batch(0.5, rows)

    # gamma and beta step with plain SGD (D7), and nothing before them differs
    assert bits(decayed.snapshot()[1][:2]) == bits(sgd.snapshot()[1][:2])
    assert to_numpy(decayed.layers[0].W).tobytes() != to_numpy(sgd.layers[0].W).tobytes()


@pytest.mark.parametrize("method", ["learn", "learn_row", "learn_batch", "learn_batch_rows"])
def test_a_one_example_training_step_is_refused_naming_the_layer(method: str, backend: Backend):
    network = _network("after a sigmoid layer", backend=backend)
    rows = _rows(3)
    before = bits(network.snapshot())

    with pytest.raises(ValueError, match=r"layer 2, BatchNorm\(activation='relu'.*D4"):
        match method:
            case "learn":
                network.learn(0.5, *rows[0])
            case "learn_row":
                network.learn_row(0.5, network.prepare_dataset(rows), 0)
            case "learn_batch":
                network.learn_batch(0.5, rows[:1])
            case _:
                network.learn_batch_rows(0.5, network.prepare_dataset(rows), [2])
    assert bits(network.snapshot()) == before


def test_snapshot_carries_the_running_averages_and_restore_returns_them(backend: Backend):
    network = _network(backend=backend)
    network.learn_batch(0.5, _rows(6))
    snapshot = network.snapshot()

    assert [len(entry) for entry in snapshot] == [1, 4, 2]
    norm = network.layers[1]
    assert bits(snapshot[1]) == bits(norm.parameters() + norm.running_state())

    network.learn_batch(0.5, _rows(6, seed=5))
    network.restore([[array.tolist() for array in entry] for entry in snapshot])  # as a loaded file
    assert bits(network.snapshot()) == bits(snapshot)


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
def test_a_checkpoint_resumes_training_by_bits(rule: UpdateRule, backend: Backend):
    network = _network("two pairs", rule, backend=backend)
    network.learn_batch(0.1, _rows(6))
    checkpoint = network.checkpoint()

    network.learn_batch(0.1, _rows(5, seed=7))
    network.learn_batch(0.1, _rows(4, seed=8))
    trained = bits(network.snapshot())

    network.restore_checkpoint(checkpoint)
    network.learn_batch(0.1, _rows(5, seed=7))
    network.learn_batch(0.1, _rows(4, seed=8))
    assert bits(network.snapshot()) == trained
