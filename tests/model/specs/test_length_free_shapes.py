"""
Length-free token networks' shapes and files (the length-generalisation workplan, stage 3, P1a): the
shape walk over token ids of any count, (None,), which only an Embedding reads and no Position
follows; format 2's {"dimension": null}; and a network's load refusing such a file until its
implementation's token layers take the count from each batch (stages 4-6). Every fixed shape walks
as before (test_layer_specs.py).
"""

import json
from pathlib import Path
from typing import Literal

import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.format2 import network_from_json
from indrajala_ml.model.persistence.format2_json import input_shape_from_json, input_to_json
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    Embedding,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    TokenMean,
)
from indrajala_ml.model.specs.spec_shapes import AnyShape, SpecShape, fixed_input_shape, spec_shapes
from indrajala_ml.model.specs.update_rules import Adam

IDS = Embedding(7, 8)
TOKEN_OUTPUT = Dense(7, output=True, activation="softmax", loss="cross_entropy")
FFN_BLOCK = Residual((LayerNorm(), Dense(16, activation="relu"), Dense(8, activation="linear", bias=True)))


def _causal(positions: Literal["none", "rotary"]) -> list[LayerSpec]:
    block = Residual((LayerNorm(), Attention(heads=2, causal=True, positions=positions)))
    return [IDS, block, FFN_BLOCK, LayerNorm(), TOKEN_OUTPUT]


LENGTH_FREE: dict[str, list[LayerSpec]] = {
    "rotary": _causal("rotary"),
    "no positions": _causal("none"),
    "an embedding, then the mean": [IDS, Residual((LayerNorm(), Attention(causal=True))), TokenMean(), TOKEN_OUTPUT],
}


def _any_count(shape: AnyShape) -> AnyShape:
    # a shape of the walk over 5 ids, with any count in place of the 5: the ids' and each token shape's
    if shape == (5,):
        return (None,)
    return (None, shape[1]) if len(shape) == 2 else shape


@pytest.mark.parametrize("specs", LENGTH_FREE.values(), ids=LENGTH_FREE.keys())
def test_token_ids_of_any_count_walk_as_a_fixed_count_with_none_for_tokens(specs: list[LayerSpec]):
    for fixed, variable in zip(spec_shapes(specs, (5,)), spec_shapes(specs, (None,)), strict=True):
        assert variable == SpecShape(_any_count(fixed.input_shape), _any_count(fixed.output_shape), fixed.positions)


def test_a_causal_transformer_gives_any_count_of_tokens_a_softmax_each():
    shapes = spec_shapes(LENGTH_FREE["rotary"], (None,))
    assert shapes[0] == SpecShape((None,), (None, 8))
    assert shapes[-1] == SpecShape((None, 8), (None, 7))


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        ([IDS, Position(), *_causal("none")[1:]], "a row per token"),
        ([Dense(5), TOKEN_OUTPUT], "read by an Embedding"),
        ([LayerNorm(), IDS, TOKEN_OUTPUT], "read by an Embedding"),
        ([Patches(2), IDS, TOKEN_OUTPUT], "read by an Embedding"),
    ],
    ids=["a position", "a dense layer", "a layer norm", "patches"],
)
def test_token_ids_of_any_count_are_read_by_an_embedding_and_followed_by_no_position(
    specs: list[LayerSpec], message: str
):
    with pytest.raises(AssertionError, match=message):
        spec_shapes(specs, (None,))


def test_a_position_over_a_fixed_count_still_walks():
    spec_shapes([IDS, Position(), *_causal("none")[1:]], (5,))


def test_format_2_records_any_count_as_a_null_dimension():
    entry = json.loads(json.dumps(input_to_json((None,), None)))
    assert entry == {"dimension": None}
    assert input_shape_from_json(entry) == (None,)
    assert input_shape_from_json(json.loads(json.dumps(input_to_json((5,), None)))) == (5,)


def test_a_fixed_input_shape_passes_and_any_count_is_refused_until_its_stage():
    assert fixed_input_shape((5,)) == (5,)
    assert fixed_input_shape((4, 4, 1)) == (4, 4, 1)
    with pytest.raises(AssertionError, match="any token count"):
        fixed_input_shape((None,))


def _length_free_file(tmp_path: Path) -> Path:
    # a fixed-count sequence network's file, its input then made any count
    network = SequentialArrayNetwork((5,), LENGTH_FREE["rotary"], Adam(), "sequence", NUMPY)
    path = tmp_path / "network.json"
    network.save(str(path))
    state = json.loads(path.read_text())
    state["input"] = input_to_json((None,), None)
    path.write_text(json.dumps(state))
    return path


def test_a_length_free_file_reads_back_and_no_network_loads_it_yet(tmp_path: Path):
    path = _length_free_file(tmp_path)
    assert network_from_json(json.loads(path.read_text())).input_shape == (None,)
    with pytest.raises(AssertionError, match="any token count"):
        load_network(str(path))
