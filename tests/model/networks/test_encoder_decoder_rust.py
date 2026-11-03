"""
The Rust encoder-decoder network (the cross-attention workplan, stage 6b, D2): numpy's passes over
the Rust backend. Its training matches numpy's within the products' rounding under the rules whose
step is linear in the gradient (numpy's exp the crate's). It lowers the loss on packed pair
windows. A one-example step is a batch of one. A saved file loads into Rust and trains on by bits,
and a numpy file's network loads into Rust as numpy's within rounding. Its seeded draws are numpy's
by bits (test_seeded_init_parity.py).
"""

from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy import attention_array_layer, token_array_layer
from indrajala_ml.model.networks.numpy.encoder_decoder_array_network import EncoderDecoderArrayNetwork
from indrajala_ml.model.networks.rust.encoder_decoder_rust_array_network import EncoderDecoderRustArrayNetwork
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.update_rules import SGD, Adam, UpdateRule
from tests.helpers import bits, exp_by_crate, learn_in_step, patching, to_numpy
from tests.model.networks import test_encoder_decoder_numpy as numpy_tests
from tests.model.networks.test_attention_rust_network import LINEAR_RULES

# numpy's softmax exps, attention's and the token-wise output's, as the crate's
attention_exp = patching(attention_array_layer, "exp", exp_by_crate)
token_exp = patching(token_array_layer, "exp", exp_by_crate)


def network(rule: UpdateRule | None = None, backend: Any = RUST, seed: int = 0) -> Any:
    cls = EncoderDecoderRustArrayNetwork if backend is RUST else EncoderDecoderArrayNetwork
    return cls.randomized(numpy_tests.spec(), numpy_tests.WINDOW, numpy_tests.WINDOW, rule or SGD(), seed=seed)


def examples() -> list[tuple[tuple[float, ...], tuple[int, ...]]]:
    return numpy_tests.batch(numpy_tests.windows())


def _assert_snapshots_close(numpy_net: Any, rust_net: Any, tolerance: float) -> None:
    # each parameter within tolerance of its layer's largest value
    for expected_entry, actual_entry in zip(numpy_net.snapshot(), rust_net.snapshot(), strict=True):
        scale = max((float(np.abs(values).max()) for values in expected_entry), default=1.0)
        for values, rust_values in zip(expected_entry, actual_entry, strict=True):
            np.testing.assert_allclose(to_numpy(rust_values), values, rtol=0.0, atol=tolerance * scale)


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
def test_training_matches_numpy_within_the_products_rounding(rule: UpdateRule):
    rust, numpy = network(rule), network(rule, backend=NUMPY)
    assert bits([to_numpy(a) for entry in rust.snapshot() for a in entry]) == bits(
        [a for entry in numpy.snapshot() for a in entry]
    )
    data = examples() * 5
    learn_in_step(0.1, data, (rust, numpy), steps=30)
    _assert_snapshots_close(numpy, rust, 1e-10)


@pytest.mark.usefixtures("attention_exp", "token_exp")
def test_one_adam_step_has_numpys_parameters_within_rounding():
    rust, numpy = network(Adam()), network(Adam(), backend=NUMPY)
    batch = examples()
    for net in (rust, numpy):
        net.learn_batch(0.01, batch)
    _assert_snapshots_close(numpy, rust, 1e-10)


def test_training_on_packed_windows_lowers_the_loss():
    rust = network(Adam())
    batch = examples()
    numpy = network(Adam(), backend=NUMPY)

    def loss() -> float:
        numpy.restore([tuple(to_numpy(a) for a in entry) for entry in rust.snapshot()])
        return numpy_tests.loss(numpy, batch)

    before = loss()
    for _ in range(40):
        rust.learn_batch(0.01, batch)
    assert loss() < 0.5 * before


def test_a_one_example_step_is_a_batch_of_one():
    one, batch = network(), network()
    state, labels = examples()[0]
    one.learn(0.1, state, labels)
    batch.learn_batch(0.1, [(state, labels)])
    assert bits([to_numpy(a) for entry in one.snapshot() for a in entry]) == bits(
        [to_numpy(a) for entry in batch.snapshot() for a in entry]
    )


def test_a_saved_file_loads_into_rust_and_trains_on_by_bits(tmp_path: Any):
    trained = network(Adam())
    batch = examples()
    trained.learn_batch(0.01, batch)
    path = str(tmp_path / "model.json")
    trained.save(path)
    loaded = load_network(path)
    assert isinstance(loaded, EncoderDecoderRustArrayNetwork)
    for net in (trained, loaded):
        net.learn_batch(0.01, batch)
    assert bits([to_numpy(a) for entry in loaded.snapshot() for a in entry]) == bits(
        [to_numpy(a) for entry in trained.snapshot() for a in entry]
    )
    state = batch[0][0]
    assert loaded.classify_state(state) == trained.classify_state(state)
