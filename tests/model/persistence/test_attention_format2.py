"""
Patch models and layer norm in format 2 and checkpoints (the layer-norm and attention workplan,
stage 5): the entries "patches", "position", "layer_norm", "attention" and "token_mean", a
token-wise dense layer's entry as any dense one's; a saved and loaded patch model, and a dense
network with flat layer norms, resume training by bits under every rule, in all three
implementations, through their class's load and load_network; the weights and optimizer state per
expanded layer (P for a position, gamma and beta for a layer norm, attention's eight arrays, none
for patches or a token mean); numpy and Rust files load into each other; and
checkpoint()/restore_checkpoint() resume by bits, also as nested lists across a worker boundary.

A multi-head patch model (the multi-head attention workplan, stage 6), two heads of 4 features over
6-wide tokens, so h·d_k isn't d, does all of the same, and its file holds attention's packed arrays.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.model.persistence.format2_json import layer_from_json, layer_to_json
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    TokenMean,
)
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import Momentum, UpdateRule
from tests.model.persistence.test_checkpoint import (
    IMPLEMENTATIONS,
    RULES,
    _network,
    _rows,
    _seeded,
    _state_bits,
    _train,
    assert_a_restored_checkpoint_resumes_by_bits,
)
from tests.model.persistence.test_format2 import SEQUENTIAL, _file, _save_and_load
from tests.model.specs.test_layer_specs import AFFINE_5, ATTENTION_BLOCK, TOKENS
from tests.saved_model_fixtures import FIXTURE_DIR, FIXTURES, fixture_class

# the README's model over a (4, 4, 1) image: 4 tokens of 4, embedded to 6, a position, the attention
# and FFN blocks, the mean, a layer norm and a softmax output
PATCH_MODEL: tuple[InputShape, list[LayerSpec]] = ((4, 4, 1), TOKENS["the README's model"])
# flat layer norms after a dropout layer and first in a residual body
FLAT_LAYER_NORM: tuple[InputShape, list[LayerSpec]] = (
    (4,),
    [
        Dense(5, dropout=0.25),
        LayerNorm(epsilon=1e-4),
        Residual((LayerNorm(), Dense(8, activation="relu"), AFFINE_5)),
        Dense(3, output=True),
    ],
)
# the README's model with two heads of 4 features over its 6-wide tokens: projections 8 wide
MULTI_HEAD_PATCH_MODEL: tuple[InputShape, list[LayerSpec]] = (
    (4, 4, 1),
    [
        Residual((LayerNorm(), Attention(heads=2, key_size=4))) if spec == ATTENTION_BLOCK else spec
        for spec in TOKENS["the README's model"]
    ],
)
ARCHITECTURES = [PATCH_MODEL, MULTI_HEAD_PATCH_MODEL, FLAT_LAYER_NORM]
ARCHITECTURE_IDS = ["patch model", "multi-head patch model", "flat layer norm"]


@pytest.mark.parametrize(
    ("spec", "entry"),
    [
        (Patches(7), {"kind": "patches", "patch_size": 7}),
        (Position(), {"kind": "position"}),
        (LayerNorm(), {"kind": "layer_norm", "epsilon": 1e-5}),
        (LayerNorm(epsilon=1e-3), {"kind": "layer_norm", "epsilon": 1e-3}),
        (Attention(), {"kind": "attention"}),
        (Attention(heads=4), {"kind": "attention", "heads": 4}),
        (Attention(key_size=8), {"kind": "attention", "key_size": 8}),
        (Attention(heads=3, key_size=8), {"kind": "attention", "heads": 3, "key_size": 8}),
        (TokenMean(), {"kind": "token_mean"}),
        (
            Residual((LayerNorm(), Attention())),
            {"kind": "residual", "body": [{"kind": "layer_norm", "epsilon": 1e-5}, {"kind": "attention"}]},
        ),
    ],
    ids=[
        "patches",
        "position",
        "layer norm",
        "layer norm epsilon",
        "attention",
        "attention heads",
        "attention key size",
        "attention heads and key size",
        "token mean",
        "attention block",
    ],
)
def test_each_new_spec_has_its_entry(spec: LayerSpec, entry: dict[str, Any]):
    assert layer_to_json(spec) == entry
    assert layer_from_json(json.loads(json.dumps(entry))) == spec


def test_a_token_wise_dense_layers_entry_is_a_dense_one():
    embed = Dense(32, activation="linear", bias=True)
    assert layer_to_json(embed) == {
        "kind": "dense",
        "size": 32,
        "activation": "linear",
        "dropout": None,
        "output": False,
        "loss": "squared",
        "bias": True,
    }


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("architecture", ARCHITECTURES, ids=ARCHITECTURE_IDS)
@pytest.mark.parametrize("loader", ["class", "load_network"])
def test_a_loaded_network_resumes_training_by_bits(
    implementation: str,
    rule: UpdateRule,
    architecture: tuple[InputShape, list[LayerSpec]],
    loader: str,
    tmp_path: Path,
):
    # train N, save, load, train M: N + M steps without the save, by bits
    input_shape, layers = architecture
    rows = _rows(input_shape, 8, seed=1)
    trained = _seeded(_network(implementation, input_shape, layers, rule), 2)
    trained.randomize()
    _train(trained, rows)

    loaded = _save_and_load(trained, tmp_path, load_network if loader == "load_network" else None)
    assert loaded.layer_specs == layers
    assert _state_bits(loaded) == _state_bits(trained)
    _train(trained, rows)
    _train(loaded, rows)

    assert _state_bits(loaded) == _state_bits(trained)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_patch_model_file_holds_an_entry_per_expanded_layer(implementation: str, tmp_path: Path):
    input_shape, layers = PATCH_MODEL
    network = _network(implementation, input_shape, layers, Momentum(0.9))
    network.randomize()
    _train(network, _rows(input_shape, 8, seed=1))

    saved = json.loads(Path(_file(network, tmp_path)).read_text())

    assert [entry["kind"] for entry in saved["layers"]] == [
        "patches",
        "dense",
        "position",
        "residual",
        "residual",
        "token_mean",
        "layer_norm",
        "dense",
    ]
    # patches, embedding, position, fork, layer norm, attention, add, fork, layer norm, ReLU,
    # affine, add, mean, layer norm, output
    weights, state = saved["weights"], saved["optimizer_state"]["layers"]
    assert len(weights) == len(state) == 15
    parameter_free = (0, 3, 6, 7, 11, 12)
    assert [weights[i] for i in parameter_free] == [[]] * len(parameter_free)
    assert all(not state[i] for i in parameter_free)  # null, or no weight sets in pure Python
    position, layer_norm, attention = 2, 4, 5
    if implementation == "python":
        # a position's row per token, its weights alone; a layer norm's ([gamma], beta) per feature;
        # attention's (weights, bias) per row of Wq, Wk, Wv and Wo
        assert [len(row) for row in weights[position]] == [1] * 4
        assert all(set(row) == {"velocity_weights"} for row in state[position])
        assert [(len(gamma), type(beta)) for gamma, beta in weights[layer_norm]] == [(1, float)] * 6
        assert [len(row) for row in weights[attention]] == [2] * 4 * 6
        assert all(set(row) == {"velocity_weights", "velocity_bias"} for row in state[attention])
    else:
        assert len(weights[position]) == 1 and set(state[position]) == {"velocity_P"}
        assert len(weights[layer_norm]) == 2 and set(state[layer_norm]) == {"velocity_gamma", "velocity_beta"}
        assert len(weights[attention]) == 8
        assert list(state[attention]) == [
            f"velocity_{name}" for name in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo")
        ]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_multi_head_files_attention_entry_holds_the_packed_arrays(implementation: str, tmp_path: Path):
    # Wq, Wk and Wv (8, 6), their biases (8,), Wo (6, 8) and bo (6,): heads as row blocks
    input_shape, layers = MULTI_HEAD_PATCH_MODEL
    network = _network(implementation, input_shape, layers, Momentum(0.9))
    network.randomize()
    _train(network, _rows(input_shape, 8, seed=1))

    saved = json.loads(Path(_file(network, tmp_path)).read_text())

    assert saved["layers"][3]["body"][1] == {"kind": "attention", "heads": 2, "key_size": 4}
    attention = saved["weights"][5]
    if implementation == "python":
        # a (weights, bias) per row: 8 of Wq, Wk and Wv each over 6 inputs, then 6 of Wo over 8
        assert [(len(weights), type(bias)) for weights, bias in attention] == [(6, float)] * 24 + [(8, float)] * 6
    else:
        shapes = [(len(array), len(array[0])) if isinstance(array[0], list) else (len(array),) for array in attention]
        assert shapes == [(8, 6), (8,), (8, 6), (8,), (8, 6), (8,), (6, 8), (6,)]


@pytest.mark.parametrize("saved_by", ["numpy", "rust"])
@pytest.mark.parametrize("architecture", ARCHITECTURES, ids=ARCHITECTURE_IDS)
def test_numpy_and_rust_files_load_into_each_other(
    saved_by: str, architecture: tuple[InputShape, list[LayerSpec]], tmp_path: Path
):
    input_shape, layers = architecture
    network = _seeded(_network(saved_by, input_shape, layers, Momentum(0.9)), 2)
    network.randomize()
    _train(network, _rows(input_shape, 8, seed=1))
    loader = SEQUENTIAL["rust" if saved_by == "numpy" else "numpy", "multiclass"]

    loaded = _save_and_load(network, tmp_path, loader.load)

    assert type(loaded) is loader
    assert _state_bits(loaded) == _state_bits(network)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("architecture", ARCHITECTURES, ids=ARCHITECTURE_IDS)
@pytest.mark.parametrize("across_workers", [False, True], ids=["in_memory", "as_lists"])
def test_a_restored_checkpoint_resumes_training_by_bits(
    implementation: str, rule: UpdateRule, architecture: tuple[InputShape, list[LayerSpec]], across_workers: bool
):
    assert_a_restored_checkpoint_resumes_by_bits(implementation, *architecture, rule, across_workers)


ATTENTION_FIXTURES = [name for name in FIXTURES if name.startswith(("Attention", "MultiHeadAttention"))]


@pytest.mark.parametrize("name", ATTENTION_FIXTURES)
def test_a_patch_model_fixture_re_saves_byte_identically(name: str, tmp_path: Path):
    # heads and key_size are written only when not the default (the multi-head attention workplan,
    # stage 2), so a one-head file is as before Attention had them; a multi-head one (stage 6) as
    # first written
    path = tmp_path / f"{name}.json"
    fixture_class(name).load(str(FIXTURE_DIR / f"{name}.json")).save(str(path))
    assert path.read_bytes() == (FIXTURE_DIR / f"{name}.json").read_bytes()


def test_there_is_a_one_head_and_a_multi_head_fixture_per_implementation():
    assert len(ATTENTION_FIXTURES) == 6


# the attention-dropout workplan's (stage 7): the sequence model with GPT's three dropouts, its
# "dropout" entries and an attention entry's "dropout" (D7)
DROPOUT_FIXTURES = [name for name in FIXTURES if name.startswith("SequenceDropout")]


@pytest.mark.parametrize("name", DROPOUT_FIXTURES)
def test_a_dropout_sequence_fixture_re_saves_byte_identically(name: str, tmp_path: Path):
    path = tmp_path / f"{name}.json"
    fixture_class(name).load(str(FIXTURE_DIR / f"{name}.json")).save(str(path))
    assert path.read_bytes() == (FIXTURE_DIR / f"{name}.json").read_bytes()
    entries = json.loads(path.read_text())["layers"]
    assert {"kind": "dropout", "p": 0.1} in entries
    assert len(DROPOUT_FIXTURES) == 3
