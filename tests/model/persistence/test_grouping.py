"""
Converting a trained model to grouped-query attention (the grouped-query attention workplan, D5,
stage 6): to_grouped on a format-2 document, any implementation. g = heads returns the document
unchanged; a converted layer's Wk, bk, Wv and bv rows are its groups' heads' means (a left fold in
head order, then divided), its Wq, bq, Wo and bo and every other layer's weights kept; numpy's,
Rust's and pure Python's documents from one seed convert to the same weights by bits; a converted
document loads into its implementation, with a fresh optimizer state, and trains; and the
refusals.
"""

import json
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.format2 import network_to_json
from indrajala_ml.model.persistence.format2_json import layer_from_json
from indrajala_ml.model.persistence.grouping import to_grouped
from indrajala_ml.model.persistence.load_network import network_from_state
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Embedding,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
    expand_specs,
)
from indrajala_ml.model.specs.update_rules import Adam
from tests.helpers import bits
from tests.model.networks import test_attention_python_network as python_networks
from tests.model.networks import test_sequence_array_network as sequences
from tests.model.networks.test_attention_array_network import IMAGE, rows
from tests.model.networks.test_attention_network import MULTI_HEAD
from tests.model.specs.test_layer_specs import FFN_BLOCK, TOKEN_OUTPUT
from tests.python_array_snapshot import as_array_snapshot, seeded_like


def _document(specs: Any, backend: Any = NUMPY) -> dict[str, Any]:
    built = sequences.network(specs, Adam()) if backend is NUMPY else _rust(specs)
    built.learn_batch(0.1, sequences.rows(specs, 4))
    return json.loads(json.dumps(network_to_json(built)))


def _rust(specs: Any) -> Any:
    built = SequentialArrayNetwork(sequences.IDS, specs, Adam(), shape="sequence", backend=RUST)
    built.rng = RUST.default_rng(3)
    built.randomize()
    return built


# the causal sequence model at four heads of 2
CAUSAL_FOUR: list[LayerSpec] = [
    Embedding(7, 6),
    Position(),
    Residual((LayerNorm(), Attention(heads=4, key_size=2, causal=True))),
    FFN_BLOCK,
    LayerNorm(),
    TOKEN_OUTPUT,
]


def _attention_index(document: dict[str, Any]) -> int:
    specs = expand_specs([layer_from_json(entry) for entry in document["layers"]])
    return next(i for i, spec in enumerate(specs) if isinstance(spec, Attention))


def test_grouping_into_the_heads_returns_the_document_unchanged():
    document = _document(CAUSAL_FOUR)
    assert to_grouped(document, 4) is document


@pytest.mark.parametrize("key_value_heads", [1, 2])
def test_a_converted_layers_key_and_value_rows_are_their_groups_heads_means(key_value_heads: int):
    document = _document(CAUSAL_FOUR)
    converted = to_grouped(document, key_value_heads)
    i = _attention_index(document)
    old, new = document["weights"][i], converted["weights"][i]
    size = 4 // key_value_heads
    for p in (2, 3, 4, 5):  # Wk, bk, Wv, bv
        heads = np.array(old[p]).reshape(4, 2, -1)  # 4 heads of d_k 2
        groups = heads.reshape(key_value_heads, size, 2, -1)
        expected = groups[:, 0]
        for m in range(1, size):
            expected = expected + groups[:, m]
        expected = (expected / size).reshape(np.array(new[p]).shape)
        assert bits(np.array(new[p])) == bits(expected), p
    for p in (0, 1, 6, 7):  # Wq, bq, Wo, bo
        assert new[p] == old[p], p
    for j, (old_layer, new_layer) in enumerate(zip(document["weights"], converted["weights"], strict=True)):
        if j != i:
            assert new_layer == old_layer, j


def test_the_converted_spec_shares_the_key_value_heads_and_the_optimizer_starts_afresh():
    converted = to_grouped(_document(CAUSAL_FOUR), 2)
    loaded = network_from_state(converted)
    attention = next(spec for spec in expand_specs(loaded.layer_specs) if isinstance(spec, Attention))
    assert attention.key_value_heads == 2
    assert converted["optimizer_state"]["t"] == 0
    assert loaded.optimizer.state().layers == {}
    # it trains
    loaded.learn_batch(0.1, sequences.rows(CAUSAL_FOUR, 4, seed=5))


@pytest.mark.parametrize("name", ["two heads", "four heads of 2, wider than d", "two layers"])
def test_numpy_rust_and_pure_python_convert_to_the_same_weights_by_bits(name: str):
    # pure Python draws numpy's weights from one seed (seeded_like); Rust's are numpy's by bits
    specs = MULTI_HEAD[name]
    python = python_networks.network(specs)
    numpy = seeded_like(python, SequentialArrayNetwork(IMAGE, specs, python.optimizer.rule, backend=NUMPY), 3)
    rust = SequentialArrayNetwork(IMAGE, specs, python.optimizer.rule, backend=RUST)
    rust.restore(numpy.snapshot())
    documents = [json.loads(json.dumps(network_to_json(built))) for built in (numpy, rust, python)]
    loaded = [network_from_state(to_grouped(document, 1)) for document in documents]
    expected = [[np.asarray(a) for a in entry] for entry in loaded[0].snapshot()]
    assert bits([[np.asarray(a.tolist()) for a in entry] for entry in loaded[1].snapshot()]) == bits(expected)
    assert bits([[np.asarray(a) for a in entry] for entry in as_array_snapshot(loaded[2])]) == bits(expected)
    # and the converted networks compute alike: numpy's and pure Python's outputs within the products' rounding
    state = rows(1)[0][0]
    np.testing.assert_allclose(
        loaded[2].predict_probabilities(state), loaded[0].predict_probabilities(state), rtol=0, atol=1e-13
    )


def test_the_refusals():
    document = _document(CAUSAL_FOUR)
    with pytest.raises(ValueError, match="doesn't divide the heads"):
        to_grouped(document, 3)
    with pytest.raises(ValueError, match="doesn't divide the heads"):
        to_grouped(document, 0)
    with pytest.raises(ValueError, match="their own key/value head"):
        to_grouped(to_grouped(document, 2), 1)
    with pytest.raises(ValueError, match="not one network's"):
        to_grouped({"format": 2, "shape": "ensemble", "classifiers": []}, 1)
    with pytest.raises(ValueError, match=r"D10"):
        to_grouped({**document, "shape": "window_recurrent"}, 1)
