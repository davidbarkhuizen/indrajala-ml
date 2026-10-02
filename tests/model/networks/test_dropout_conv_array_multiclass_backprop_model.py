from typing import Any

import pytest

from indrajala_ml.model.networks.numpy.dropout_conv_vectorized_multiclass_backprop_classifier_network import (
    DropoutConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.rust.dropout_conv_rust_array_multiclass_backprop_classifier_network import (
    DropoutConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.update_rules import SGD, UpdateRule
from tests.array_network_contract import (
    CLASS_COUNT,
    CONV_DENSE_LAYER_SIZES,
    CONV_SIDE,
    CONV_SPECS,
    ArrayNetworkSpec,
    conv_network_tests,
)
from tests.helpers import Backend

DROP_PROBABILITY = 0.5
NetworkCls = (
    type[DropoutConvVectorizedMultiClassBackpropClassifierNetwork]
    | type[DropoutConvRustArrayMultiClassBackpropClassifierNetwork]
)


# dropout does nothing at inference, the only state in which the array networks can be compared
# with the per-node reference, whose masks come from Python's random; the Sequential network of
# the same specs is compared in training, seeded alike
def _equivalent(sizes: list[int], output: int, drop_probability: float) -> tuple[list[LayerSpec], UpdateRule]:
    # this network's dense layers as layer specs, and its update rule
    return (
        [*(Dense(size, dropout=drop_probability) for size in sizes), Dense(output, output=True)],
        SGD(),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": DropoutConvVectorizedMultiClassBackpropClassifierNetwork,
        "rust": DropoutConvRustArrayMultiClassBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    hyperparameters={"drop_probability": DROP_PROBABILITY},
    parity_in_training=False,
    saved_hyperparameters_test="drop_probability",
    saved_hyperparameters={"drop_probability": 0.3},
    invalid_hyperparameters={"drop_probability": 1.0},
)

globals().update(conv_network_tests(SPEC))


@pytest.fixture
def network_cls(backend: Backend) -> NetworkCls:
    return SPEC.network_cls[backend.name]


def _network(network_cls: NetworkCls) -> Any:
    return network_cls.randomized(CONV_SIDE, CONV_SIDE, CONV_SPECS, [6, 5], CLASS_COUNT, DROP_PROBABILITY, seed=0)


def test_only_the_dense_hidden_layers_drop_out(network_cls: NetworkCls):
    network = _network(network_cls)

    assert network.hidden_layers == network.layers[len(CONV_SPECS) : -1]
    assert len(network.hidden_layers) == 2
    assert network._training_mode_layers == network.hidden_layers


def test_learn_batch_masks_the_dense_hidden_layers_and_leaves_training_mode_off(network_cls: NetworkCls):
    network = _network(network_cls)
    state = tuple(0.01 * i for i in range(CONV_SIDE * CONV_SIDE))
    batch = [(state, j % CLASS_COUNT) for j in range(4)]

    network.learn_batch(0.1, batch)

    assert all(not layer.training for layer in network.hidden_layers)
    for layer in network.hidden_layers:
        rows = layer._mask_batch.tolist()
        assert (len(rows), len(rows[0])) == (len(batch), layer.size)


def test_predict_probabilities_is_deterministic_at_inference(network_cls: NetworkCls):
    network = _network(network_cls)
    state = tuple(0.01 * i for i in range(CONV_SIDE * CONV_SIDE))

    network.learn(0.1, state, 1)

    assert len({tuple(network.predict_probabilities(state)) for _ in range(10)}) == 1


def test_drop_probability_is_a_required_constructor_argument(network_cls: NetworkCls):
    with pytest.raises(TypeError):
        network_cls(CONV_SIDE, CONV_SIDE, CONV_SPECS, CONV_DENSE_LAYER_SIZES, CLASS_COUNT)  # type: ignore[call-arg]
