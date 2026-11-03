"""
The pure-Python encoder-decoder network (the cross-attention workplan, stage 7b, D2): numpy's draws
from the same seed by bits, its training numpy's within the products' rounding (numpy's exps
math.exp), with dropout too, whose masks are numpy's. A one-example step is a batch of one, and a
saved file loads into pure Python and trains on by bits. With test_encoder_decoder_rust.py's, the
three implementations agree.
"""

from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.numpy import attention_array_layer, token_array_layer
from indrajala_ml.model.networks.numpy.encoder_decoder_array_network import EncoderDecoderArrayNetwork
from indrajala_ml.model.networks.python.encoder_decoder_backprop_network import EncoderDecoderBackpropNetwork
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from indrajala_ml.pcg64 import generator_state
from tests.helpers import exp_by_math, patching
from tests.model.networks import test_encoder_decoder_numpy as numpy_tests
from tests.python_array_snapshot import as_array_snapshot, assert_seeded_alike

# numpy's softmax exps, attention's and the token-wise output's, as math.exp
attention_exp = patching(attention_array_layer, "exp", exp_by_math)
token_exp = patching(token_array_layer, "exp", exp_by_math)


def networks(rule: UpdateRule, dropout: float = 0.0, seed: int = 0) -> tuple[Any, Any]:
    spec, window = numpy_tests.spec(dropout=dropout), numpy_tests.WINDOW
    python = EncoderDecoderBackpropNetwork.randomized(spec, window, window, rule, seed=seed)
    numpy = EncoderDecoderArrayNetwork.randomized(spec, window, window, rule, seed=seed)
    return python, numpy


def examples() -> list[tuple[tuple[float, ...], tuple[int, ...]]]:
    return numpy_tests.batch(numpy_tests.windows())


def assert_close(python: Any, numpy: Any, tolerance: float = 1e-10) -> None:
    for expected_entry, actual_entry in zip(numpy.snapshot(), as_array_snapshot(python), strict=True):
        scale = max((float(np.abs(values).max()) for values in expected_entry), default=1.0)
        for values, python_values in zip(expected_entry, actual_entry, strict=True):
            np.testing.assert_allclose(np.asarray(python_values), values, rtol=0.0, atol=tolerance * scale)


@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("seed", [0, 42])
def test_randomize_draws_numpys_network_by_bits(seed: int, dropout: float):
    assert_seeded_alike(*networks(SGD(), dropout, seed))


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9)], ids=lambda rule: type(rule).__name__)
def test_training_matches_numpy_within_the_products_rounding(rule: UpdateRule):
    python, numpy = networks(rule)
    batch = examples()
    for _ in range(3):
        for net in (python, numpy):
            net.learn_batch(0.1, batch)
    assert_close(python, numpy)


@pytest.mark.usefixtures("attention_exp", "token_exp")
def test_a_dropping_step_draws_numpys_masks_and_matches_within_rounding():
    python, numpy = networks(SGD(), dropout=0.1, seed=3)
    batch = examples()
    for net in (python, numpy):
        net.learn_batch(0.1, batch)
    assert_close(python, numpy)
    # as many draws: the generators where numpy's draws leave numpy's
    assert generator_state(python.rng) == generator_state(numpy.rng)


def test_a_one_example_step_is_a_batch_of_one():
    one, batch = networks(SGD())[0], networks(SGD())[0]
    state, labels = examples()[0]
    one.learn(0.1, state, labels)
    batch.learn_batch(0.1, [(state, labels)])
    assert one.snapshot() == batch.snapshot()


def test_a_saved_file_loads_into_python_and_trains_on_by_bits(tmp_path: Any):
    trained, _ = networks(Adam())
    batch = examples()[:3]
    trained.learn_batch(0.01, batch)
    path = str(tmp_path / "model.json")
    trained.save(path)
    loaded = load_network(path)
    assert isinstance(loaded, EncoderDecoderBackpropNetwork)
    for net in (trained, loaded):
        net.learn_batch(0.01, batch)
    assert loaded.snapshot() == trained.snapshot()
    assert loaded.classify_state(batch[0][0]) == trained.classify_state(batch[0][0])
