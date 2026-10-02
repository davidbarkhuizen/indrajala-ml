import pytest

from indrajala_ml.model.networks.numpy.dropout_array_backprop_classifier_network import (
    DropoutArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.rust.dropout_rust_array_backprop_classifier_network import (
    DropoutRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.update_rules import SGD, UpdateRule
from tests.array_network_contract import DIMENSION, LAYER_SIZES, ArrayNetworkSpec, single_output_network_tests
from tests.helpers import Backend

DROP_PROBABILITY = 0.5
NetworkCls = type[DropoutArrayBackpropClassifierNetwork] | type[DropoutRustArrayBackpropClassifierNetwork]


# dropout does nothing at inference, the only state in which the array networks can be compared
# with the per-node reference, whose masks come from Python's random
def _equivalent(sizes: list[int], output: int, drop_probability: float) -> tuple[list[LayerSpec], UpdateRule]:
    # this network as layer specs and an update rule
    return (
        [*(Dense(size, dropout=drop_probability) for size in sizes), Dense(output, output=True)],
        SGD(),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": DropoutArrayBackpropClassifierNetwork,
        "rust": DropoutRustArrayBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    hyperparameters={"drop_probability": DROP_PROBABILITY},
    parity_in_training=False,
    saved_hyperparameters_test="drop_probability",
    saved_hyperparameters={"drop_probability": 0.3},
    invalid_hyperparameters={"drop_probability": 1.0},
)

globals().update(single_output_network_tests(SPEC))


@pytest.fixture
def network_cls(backend: Backend) -> NetworkCls:
    return SPEC.network_cls[backend.name]


def test_predict_probability_is_deterministic_run_to_run_no_stochasticity_at_inference(network_cls: NetworkCls):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION, drop_probability=DROP_PROBABILITY)
    state = tuple(0.1 * i for i in range(DIMENSION))

    assert len({network.predict_probability(state) for _ in range(20)}) == 1


def test_learn_and_learn_batch_leave_training_mode_off(network_cls: NetworkCls):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION, drop_probability=DROP_PROBABILITY)
    batch = [(tuple(0.1 * i + 0.01 * j for i in range(DIMENSION)), float(j % 2)) for j in range(6)]

    network.learn(0.1, batch[0][0], batch[0][1])
    assert all(not layer.training for layer in network.hidden_layers)

    network.learn_batch(0.1, batch)
    assert all(not layer.training for layer in network.hidden_layers)
    for layer in network.hidden_layers:
        rows = layer._mask_batch.tolist()
        assert (len(rows), len(rows[0])) == (len(batch), layer.size)


def test_drop_probability_is_a_required_constructor_argument(network_cls: NetworkCls):

    with pytest.raises(TypeError):
        network_cls(LAYER_SIZES, DIMENSION)  # type: ignore[call-arg]
