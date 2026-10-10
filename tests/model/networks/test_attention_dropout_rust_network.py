"""
Dropout among the tokens on Rust (the attention-dropout workplan, stage 5): every dropping layer's
mask against numpy's by bits from one seed, and the generator left where numpy's is; the token
dropout against numpy's by bits (no product rounds); inference drawing nothing, as the network
without dropout by bits; learn against a batch of one; save and load by bits; and parity with
numpy, after 50 steps under the rules whose step is linear in the gradient and per step under Adam,
both sides drawing the same masks. The cases are tests/model/specs/test_layer_specs.py's DROPOUT;
the gradient check is tests/model/networks/test_attention_dropout_array_network.py's, on numpy.
"""

from pathlib import Path
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.layers.numpy.token_array_layer import TokenDropoutArrayLayer
from indrajala_ml.model.layers.rust.attention_rust_array_layer import AttentionRustArrayLayer
from indrajala_ml.model.layers.rust.token_rust_array_layer import TokenDropoutRustArrayLayer
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from tests.helpers import assert_learn_and_a_batch_of_one_agree, bits, to_numpy
from tests.model.networks.test_attention_dropout_array_network import GPT, _without_dropout
from tests.model.networks.test_attention_rust_network import (
    LINEAR_RULES,
    _assert_every_adam_step_has_numpys_gradients,  # pyright: ignore[reportPrivateUsage]
    _assert_training_matches_numpy,  # pyright: ignore[reportPrivateUsage]
)
from tests.model.networks.test_sequence_array_network import rows
from tests.model.networks.test_sequence_rust_network import attention_exp, network, token_exp
from tests.model.specs.test_layer_specs import DROPOUT

__all__ = ["attention_exp", "token_exp"]  # the fixtures, used by name


def _masks(built: Any) -> list[Any]:
    # each dropping layer's last mask (an attention at dropout 0 draws none), numpy's layout: attention's (N, h, T, T), the token dropout's
    # (N, T * d); a Rust attention's mask is packed as its P, (N * T, h * T)
    masks: list[Any] = []
    for layer in built.layers:
        if isinstance(layer, TokenDropoutArrayLayer):
            masks.append(layer._mask_batch)  # pyright: ignore[reportPrivateUsage]
        elif isinstance(layer, TokenDropoutRustArrayLayer):
            masks.append(to_numpy(layer._mask))  # pyright: ignore[reportPrivateUsage, reportArgumentType]
        elif isinstance(layer, AttentionArrayLayer) and layer._M is not None:  # pyright: ignore[reportPrivateUsage]
            masks.append(layer._M)  # pyright: ignore[reportPrivateUsage]
        elif isinstance(layer, AttentionRustArrayLayer) and layer._M is not None:  # pyright: ignore[reportPrivateUsage]
            packed = to_numpy(layer._M)  # pyright: ignore[reportPrivateUsage]
            n, h, t = packed.shape[0] // layer.tokens, layer.heads, layer.tokens
            masks.append(packed.reshape(n, t, h, t).transpose(0, 2, 1, 3))
    return masks


@pytest.mark.parametrize("name", DROPOUT)
def test_every_mask_is_numpys_by_bits_from_one_seed(name: str):
    specs = DROPOUT[name]
    rust, numpy = network(specs), network(specs, backend=NUMPY)
    data = rows(specs, 3)
    for seed in (9, 10):
        rust.rng, numpy.rng = RUST.default_rng(seed), NUMPY.default_rng(seed)
        rust.learn_batch(0.1, data)
        numpy.learn_batch(0.1, data)
        assert _masks(rust) and bits(_masks(rust)) == bits(_masks(numpy))
        assert rust.rng.state == numpy.rng.bit_generator.state


def test_the_token_dropout_is_numpys_by_bits():
    numpy_layer, rust_layer = TokenDropoutArrayLayer(4, 6, 0.25), TokenDropoutRustArrayLayer(4, 6, 0.25)
    numpy_layer.set_rng(NUMPY.default_rng(5))
    rust_layer.set_rng(RUST.default_rng(5))
    X = np.random.default_rng(1).uniform(-1.0, 1.0, (3, 24))
    delta = np.random.default_rng(2).uniform(-1.0, 1.0, (3, 24))
    for training in (True, False):
        numpy_layer.set_training_mode(training)
        rust_layer.set_training_mode(training)
        expected = numpy_layer.forward_batch(X)
        assert bits(to_numpy(rust_layer.forward_batch(pa.Array(X.tolist())))) == bits(expected)
        numpy_layer._backward(delta)  # pyright: ignore[reportPrivateUsage]
        rust_layer.delta_batch = pa.Array(delta.tolist())
        assert bits(to_numpy(rust_layer.downstream_batch())) == bits(numpy_layer.downstream_batch())


def test_the_token_dropouts_single_example_is_its_batch_of_one():
    x = np.random.default_rng(1).uniform(-1.0, 1.0, 24)
    one, batch = TokenDropoutRustArrayLayer(4, 6, 0.25), TokenDropoutRustArrayLayer(4, 6, 0.25)
    for layer in (one, batch):
        layer.set_rng(RUST.default_rng(5))
        layer.set_training_mode(True)
    single = to_numpy(one.forward(pa.Array(x.tolist())))
    assert bits(single) == bits(to_numpy(batch.forward_batch(pa.Array([x.tolist()])))[0])


def test_inference_draws_nothing_and_is_the_network_without_dropout_by_bits():
    built, plain = network(GPT), network(_without_dropout(GPT))
    built.learn_batch(0.5, rows(GPT, 4, seed=2))
    plain.restore(built.snapshot())
    before = built.rng.state
    prepared = built.prepare_dataset(rows(GPT, 5))
    assert bits([to_numpy(X) for X in built.forward_rows(prepared)]) == bits(
        [to_numpy(X) for X in plain.forward_rows(prepared)]
    )
    assert built.rng.state == before


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=["sgd", "momentum", "adam"])
@pytest.mark.parametrize("name", DROPOUT)
def test_learn_and_a_batch_of_one_agree(name: str, rule: UpdateRule):
    specs = DROPOUT[name]
    assert_learn_and_a_batch_of_one_agree("rust", network(specs, rule), network(specs, rule), rows(specs, 4))


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
    assert bits([to_numpy(a) for entry in loaded.snapshot() for a in entry]) == bits(
        [to_numpy(a) for entry in built.snapshot() for a in entry]
    )


# parity with numpy: both networks' generators start alike, so every step drops the same units


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", DROPOUT)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    _assert_training_matches_numpy(DROPOUT[name], rule, 0.1, network, rows)


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("name", DROPOUT)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    _assert_every_adam_step_has_numpys_gradients(DROPOUT[name], 0.1, network, rows)
