import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.dropout_rust_array_multiclass_backprop_classifier_network import (
    DropoutRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.dropout_vectorized_multiclass_backprop_classifier_network import (
    DropoutVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from tests.array_network_contract import (
    CLASS_COUNT,
    DIMENSION,
    LAYER_SIZES,
    ArrayNetworkSpec,
    multiclass_network_tests,
)
from tests.helpers import Backend, matching_dropout_array_backprop_networks

DROP_PROBABILITY = 0.5
NetworkCls = (
    type[DropoutVectorizedMultiClassBackpropClassifierNetwork]
    | type[DropoutRustArrayMultiClassBackpropClassifierNetwork]
)

# dropout does nothing at inference, the only state in which the array networks can be compared
# with the per-node reference, whose masks come from Python's random (numpy and Rust are compared
# with each other in training below)
SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": DropoutVectorizedMultiClassBackpropClassifierNetwork,
        "rust": DropoutRustArrayMultiClassBackpropClassifierNetwork,
    },
    matching=matching_dropout_array_backprop_networks,
    hyperparameters={"drop_probability": DROP_PROBABILITY},
    parity_in_training=False,
    saved_hyperparameters_test="drop_probability",
    saved_hyperparameters={"drop_probability": 0.3},
    invalid_hyperparameters={"drop_probability": 1.0},
)

globals().update(multiclass_network_tests(SPEC))


@pytest.fixture
def network_cls(backend: Backend) -> NetworkCls:
    return SPEC.network_cls[backend.name]


def test_predict_probabilities_is_deterministic_run_to_run_no_stochasticity_at_inference(network_cls: NetworkCls):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY)
    state = tuple(0.1 * i for i in range(DIMENSION))

    predictions = [tuple(network.predict_probabilities(state)) for _ in range(20)]

    assert len(set(predictions)) == 1


def test_predict_probabilities_between_learn_calls_is_unaffected_by_training_mode(network_cls: NetworkCls):

    # learn() switches training on for its forward pass only; a prediction after it is at eval
    network = network_cls.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY)
    state = tuple(0.1 * i for i in range(DIMENSION))

    network.learn(0.1, state, 1)

    assert all(not layer.training for layer in network.hidden_layers)
    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == CLASS_COUNT
    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_learn_batch_draws_an_independent_mask_per_example_and_leaves_training_mode_off(network_cls: NetworkCls):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY)
    batch = [(tuple(0.1 * i + 0.01 * j for i in range(DIMENSION)), j % CLASS_COUNT) for j in range(6)]

    network.learn_batch(0.1, batch)

    assert all(not layer.training for layer in network.hidden_layers)
    for layer in network.hidden_layers:
        rows = layer._mask_batch.tolist()
        assert (len(rows), len(rows[0])) == (len(batch), layer.size)


def test_learn_moves_the_weights(network_cls: NetworkCls):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY)
    before = network.snapshot()

    for i in range(10):
        state = tuple(0.1 * i + 0.01 * j for j in range(DIMENSION))
        network.learn(0.1, state, i % CLASS_COUNT)

    after = network.snapshot()
    assert any(not np.allclose(W1.tolist(), W2.tolist()) for (W1, _b1), (W2, _b2) in zip(before, after))


def test_drop_probability_is_a_required_constructor_argument(network_cls: NetworkCls):

    with pytest.raises(TypeError):
        network_cls(LAYER_SIZES, DIMENSION, CLASS_COUNT)  # type: ignore[call-arg]


# the no-dropout pair is the control: the backends' matmuls already differ by an ULP or so, and
# dropout must add nothing to that
END_TO_END_PAIRS = {
    "dropout": (
        DropoutVectorizedMultiClassBackpropClassifierNetwork,
        DropoutRustArrayMultiClassBackpropClassifierNetwork,
    ),
    "no-dropout control": (
        VectorizedMultiClassBackpropClassifierNetwork,
        RustArrayMultiClassBackpropClassifierNetwork,
    ),
}
END_TO_END_WEIGHT_ATOL = 1e-13  # the conv networks' numpy-vs-Rust bar (test_conv_array_multiclass_backprop_model.py)


def _masks(network: Any) -> list[Any]:
    return [layer._mask_batch.tolist() for layer in getattr(network, "hidden_layers", [])]


@pytest.mark.parametrize("pair", END_TO_END_PAIRS)
@pytest.mark.parametrize("seed", [0, 1])
def test_numpy_and_rust_train_alike_when_seeded_alike(pair: str, seed: int):
    # seeded alike, both backends start from the same weights and draw the same masks, so a few
    # epochs of learn_batch and learn keep them together, checked after every step
    numpy_cls, rust_cls = END_TO_END_PAIRS[pair]
    extra = (DROP_PROBABILITY,) if pair == "dropout" else ()
    rng = random.Random(seed)
    rows = [(tuple(rng.uniform(-1.0, 1.0) for _ in range(12)), rng.randrange(CLASS_COUNT)) for _ in range(48)]

    NUMPY.seed(seed)
    numpy_network = numpy_cls.randomized([16, 8], 12, CLASS_COUNT, *extra)
    RUST.seed(seed)
    rust_network = rust_cls.randomized([16, 8], 12, CLASS_COUNT, *extra)

    for _epoch in range(4):
        for start in range(0, len(rows), 8):
            numpy_network.learn_batch(0.5, rows[start : start + 8])
            rust_network.learn_batch(0.5, rows[start : start + 8])
            assert _masks(rust_network) == _masks(numpy_network)
            _assert_snapshots_close(numpy_network, rust_network)
        for state, label in rows[:6]:
            numpy_network.learn(0.5, state, label)
            rust_network.learn(0.5, state, label)
            _assert_snapshots_close(numpy_network, rust_network)


def _assert_snapshots_close(numpy_network: Any, rust_network: Any) -> None:
    for numpy_entry, rust_entry in zip(numpy_network.snapshot(), rust_network.snapshot()):
        for numpy_array, rust_array in zip(numpy_entry, rust_entry):
            np.testing.assert_allclose(rust_array.tolist(), numpy_array, rtol=0, atol=END_TO_END_WEIGHT_ATOL)
