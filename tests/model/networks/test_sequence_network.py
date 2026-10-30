"""
Sequence models on each implementation, numpy, Rust and pure Python (the DRY third pass workplan,
D10): learn against a batch of one, for the sequence, rotary and attention-dropout models, and a
saved rotary or dropout model training on by bits. The layers, and what one implementation alone
has (the exact tests, the parity tests, each file's own fields), stay in
tests/model/networks/test_{sequence,rotary,attention_dropout}_{array,rust,python}_network.py; this
module reuses their cases.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import LayerSpec
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from tests.helpers import Implementation, assert_learn_and_a_batch_of_one_agree, bits
from tests.model.networks import test_sequence_array_network as array
from tests.model.networks import test_sequence_python_network as python
from tests.model.networks import test_sequence_rust_network as rust
from tests.model.networks.test_sequence_array_network import rows
from tests.model.specs.test_layer_specs import DROPOUT, ROTARY, SEQUENCE

NETWORK: dict[str, Callable[..., Any]] = {"numpy": array.network, "rust": rust.network, "python": python.network}
LEARNED = {"sequence": SEQUENCE, "rotary": ROTARY, "attention dropout": DROPOUT}
# a rotary model's table derives from its spec, so it isn't saved (the rotary workplan, D4); a dropout
# model's generator state is saved, so its masks resume by bits (the attention-dropout workplan, D7)
SAVED = {"rotary": ROTARY, "attention dropout": DROPOUT}


def _cases(kinds: dict[str, dict[str, list[LayerSpec]]]) -> list[Any]:
    return [pytest.param(kind, name, id=f"{kind}: {name}") for kind, models in kinds.items() for name in models]


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=["sgd", "momentum", "adam"])
@pytest.mark.parametrize(("kind", "name"), _cases(LEARNED))
def test_learn_and_a_batch_of_one_agree(kind: str, name: str, rule: UpdateRule, implementation: Implementation):
    # a single example's dropout masks are a batch of one's, from the same generator
    specs = LEARNED[kind][name]
    build = NETWORK[implementation]
    assert_learn_and_a_batch_of_one_agree(implementation, build(specs, rule), build(specs, rule), rows(specs, 4))


@pytest.mark.parametrize(("kind", "name"), _cases(SAVED))
def test_a_saved_rotary_or_dropout_network_loads_and_trains_on_by_bits(
    kind: str, name: str, implementation: Implementation, tmp_path: Path
):
    specs = SAVED[kind][name]
    built = NETWORK[implementation](specs, Adam())
    data = rows(specs, 6)
    built.learn_batch(0.1, data[:3])
    path = str(tmp_path / "network.json")
    built.save(path)
    loaded = load_network(path)
    assert loaded.layer_specs == specs
    built.learn_batch(0.1, data[3:])
    loaded.learn_batch(0.1, data[3:])
    assert bits(loaded.snapshot()) == bits(built.snapshot())
