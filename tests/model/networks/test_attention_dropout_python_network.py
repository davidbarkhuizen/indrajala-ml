"""
Dropout among the tokens in pure Python (the attention-dropout workplan, stage 6, D6): every dropping
layer's masks, pass by pass, against numpy's by bits from one seed, and the generator left where
numpy's is; the token dropout against numpy's by bits (no product rounds; a dropped negative is
-0.0, as numpy's); attention at dropout 0 drawing nothing and leaving the network example-major;
inference drawing nothing, as the network without dropout by bits; learn against a batch of one;
save and load by bits; and parity with numpy, after 50 steps under the rules whose step is linear
in the gradient and per step under Adam, both sides drawing the same masks. The cases are
tests/model/specs/test_layer_specs.py's DROPOUT; numpy's exps are math.exp, as pure Python's.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.numpy.token_array_layer import TokenDropoutArrayLayer
from indrajala_ml.model.layers.python.attention_layer import AttentionLayer
from indrajala_ml.model.layers.python.state_layer import StateLayer
from indrajala_ml.model.layers.python.token_layer import TokenDropoutLayer
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from indrajala_ml.pcg64 import default_rng, generator_state
from tests.helpers import assert_learn_and_a_batch_of_one_agree, assert_snapshots_close, bits, learn_in_step
from tests.model.networks.test_attention_dropout_array_network import GPT, _without_dropout
from tests.model.networks.test_attention_dropout_rust_network import _masks  # pyright: ignore[reportPrivateUsage]
from tests.model.networks.test_attention_python_network import LINEAR_RULES, assert_every_step_has_numpys_gradients
from tests.model.networks.test_sequence_array_network import rows
from tests.model.networks.test_sequence_python_network import attention_exp, network, numpy_network, token_exp
from tests.model.specs.test_layer_specs import DROPOUT, SEQUENCE
from tests.python_array_snapshot import as_array_snapshot, assert_seeded_alike

__all__ = ["attention_exp", "token_exp"]  # the fixtures, used by name


def _record_masks(built: Any) -> list[list[Any]]:
    # each dropping layer's mask per forward pass, in call order: the layer-major path runs a layer
    # for every example before the next layer, so a layer's passes are its batch's rows
    masks: list[list[Any]] = []
    for layer in built.trainable_layers:
        if not (isinstance(layer, TokenDropoutLayer) or (isinstance(layer, AttentionLayer) and layer.draws)):
            continue
        passes: list[Any] = []
        masks.append(passes)

        def forward(layer: Any = layer, passes: list[Any] = passes, inner: Any = layer.forward) -> None:
            inner()
            if isinstance(layer, TokenDropoutLayer):
                passes.append([node._mask for node in layer.nodes])  # pyright: ignore[reportPrivateUsage]
            else:
                # (T, h * T), head i in columns i * T.., as numpy's (h, T, T)
                packed, t = np.array(layer._M), layer.tokens  # pyright: ignore[reportPrivateUsage]
                passes.append(packed.reshape(t, layer.heads, t).transpose(1, 0, 2))

        layer.forward = forward
    return masks


@pytest.mark.parametrize("name", DROPOUT)
def test_every_mask_is_numpys_by_bits_from_one_seed(name: str):
    specs = DROPOUT[name]
    python, array = network(specs), numpy_network(specs)
    assert_seeded_alike(python, array)
    data = rows(specs, 3)
    masks = _record_masks(python)
    for seed in (9, 10):
        for passes in masks:
            passes.clear()
        python.rng, array.rng = default_rng(seed), np.random.default_rng(seed)
        python.learn_batch(0.1, data)
        array.learn_batch(0.1, data)
        assert masks and bits([np.array(passes) for passes in masks]) == bits(_masks(array))
        assert generator_state(python.rng) == array.rng.bit_generator.state


def test_the_token_dropout_is_numpys_by_bits_a_dropped_negative_included():
    values = np.random.default_rng(1).uniform(-1.0, 1.0, 24)
    delta = np.random.default_rng(2).uniform(-1.0, 1.0, 24)
    state = StateLayer(24, [(-1.0, 1.0)] * 24)
    state.update_state(tuple(float(v) for v in values))
    python = TokenDropoutLayer(state, 4, 6, 0.25)
    array = TokenDropoutArrayLayer(4, 6, 0.25)
    python.set_rng(default_rng(5))
    array.set_rng(np.random.default_rng(5))
    for training in (True, False):
        python.set_training_mode(training)
        array.set_training_mode(training)
        python.forward()
        expected = array.forward_batch(values[np.newaxis, :])[0]
        assert bits([node.value() for node in python.nodes]) == bits(expected)
        if training:
            # a dropped negative is -0.0 on both sides, and the check reaches one
            dropped = [x < 0.0 and node._mask == 0.0 for x, node in zip(values, python.nodes)]  # pyright: ignore[reportPrivateUsage]
            assert any(dropped) and all(np.signbit(expected[dropped]))
        for node, d in zip(python.nodes, delta):
            node.delta = float(d)
        array._backward(delta[np.newaxis, :])  # pyright: ignore[reportPrivateUsage]
        assert bits([python.downstream_sum(i) for i in range(24)]) == bits(array.downstream_batch()[0])


def test_attention_at_dropout_0_draws_nothing_and_the_network_stays_example_major():
    built = network(SEQUENCE["a causal transformer"])
    assert not built._layer_major  # pyright: ignore[reportPrivateUsage]
    before = generator_state(built.rng)
    built.learn_batch(0.1, rows(SEQUENCE["a causal transformer"], 3))
    assert generator_state(built.rng) == before
    assert network(GPT)._layer_major  # pyright: ignore[reportPrivateUsage]


def test_inference_draws_nothing_and_is_the_network_without_dropout_by_bits():
    built, plain = network(GPT), network(_without_dropout(GPT))
    built.learn_batch(0.5, rows(GPT, 4, seed=2))
    plain.restore(built.snapshot())
    before = generator_state(built.rng)
    states = [state for state, _ in rows(GPT, 5)]
    assert bits([built._forward_outputs(s) for s in states]) == bits(  # pyright: ignore[reportPrivateUsage]
        [plain._forward_outputs(s) for s in states]  # pyright: ignore[reportPrivateUsage]
    )
    assert generator_state(built.rng) == before


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=["sgd", "momentum", "adam"])
@pytest.mark.parametrize("name", DROPOUT)
def test_learn_and_a_batch_of_one_agree(name: str, rule: UpdateRule):
    # learn's example-major pass draws a batch of one's masks, layer by layer
    specs = DROPOUT[name]
    assert_learn_and_a_batch_of_one_agree("python", network(specs, rule), network(specs, rule), rows(specs, 4))


@pytest.mark.parametrize("name", DROPOUT)
def test_a_saved_dropout_network_loads_and_trains_on_by_bits(name: str, tmp_path: Path):
    specs = DROPOUT[name]
    built = network(specs, Adam())
    data = rows(specs, 6)
    built.learn_batch(0.1, data[:3])
    path = str(tmp_path / "dropout.json")
    built.save(path)
    loaded = load_network(path)
    assert loaded.layer_specs == specs
    built.learn_batch(0.1, data[3:])
    loaded.learn_batch(0.1, data[3:])
    assert bits(loaded.snapshot()) == bits(built.snapshot())


# parity with numpy: both networks' generators start alike, so every step drops the same units


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", DROPOUT)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    specs = DROPOUT[name]
    python, array = network(specs, rule), numpy_network(specs, rule)
    assert_seeded_alike(python, array)
    learn_in_step(0.1, rows(specs, 40), (python, array))
    assert generator_state(python.rng) == array.rng.bit_generator.state
    assert_snapshots_close(as_array_snapshot(python), array.snapshot())


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("name", DROPOUT)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    specs = DROPOUT[name]
    assert_every_step_has_numpys_gradients(network(specs, Adam()), numpy_network(specs), rows(specs, 40))
