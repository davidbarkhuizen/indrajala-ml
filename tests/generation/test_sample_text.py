"""
Sampling text (the generation workplan, D5, D6, D8, stage 7): the tokenizer file, generate's loop
and its uncached twin, and scripts/sample_text.py's train and sample end to end on a smoke run.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from indrajala_ml.generation.generate import cached, generate, uncached
from indrajala_ml.generation.sampling import Sampler
from indrajala_ml.generation.tokenizer import CharacterTokenizer, tokenizer_path
from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import Attention, Dense, Embedding, LayerNorm, Residual
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.pcg64 import default_rng

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "generation"))
import sample_text  # scripts/ isn't a package

WINDOW = 6


def _network() -> object:
    specs = [
        Embedding(5, 8),
        Residual((LayerNorm(), Attention(heads=2, causal=True, positions="rotary"))),
        LayerNorm(),
        Dense(5, output=True, activation="softmax", loss="cross_entropy"),
    ]
    network = SequentialArrayNetwork((WINDOW,), specs, Adam(), "sequence", NUMPY)
    network.rng = np.random.default_rng(0)
    network.randomize()
    return network


def test_the_tokenizer_file_round_trips_beside_its_model(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer.of("toy", "abcab\n")
    path = tokenizer_path(tmp_path / "m.json")
    assert path.name == "m.tokenizer.json"
    tokenizer.save(path)
    loaded = CharacterTokenizer.load(path)
    assert loaded == tokenizer
    assert loaded.symbols == "\nabc"
    assert loaded.decode(loaded.encode("cab")) == "cab"


def test_the_tokenizer_refuses_an_unknown_character_and_an_unknown_kind(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer.of("toy", "abc")
    with pytest.raises(ValueError, match="no 'z'"):
        tokenizer.encode("az")
    path = tmp_path / "t.json"
    path.write_text('{"kind": "bpe", "corpus": "x", "sha256": "", "symbols": "a"}')
    with pytest.raises(ValueError, match="unknown tokenizer kind"):
        CharacterTokenizer.load(path)


def test_generate_draws_then_steps_each_token_but_the_last() -> None:
    calls: list[object] = []

    def prefill(ids: Sequence[int]) -> list[float]:
        calls.append(("prefill", list(ids)))
        return [0.0, 1.0]

    def step(token: int) -> list[float]:
        calls.append(("step", token))
        return [1.0, 0.0]

    ids = generate(prefill, step, [0], 3, Sampler(), default_rng(0))
    assert ids == [1, 0, 0]
    assert calls == [("prefill", [0]), ("step", 1), ("step", 0)]


def test_cached_and_uncached_sample_the_same_text_within_the_window() -> None:
    network = _network()
    greedy = Sampler(temperature=0.0)
    prompt = [1, 2]
    a = generate(*cached(network), prompt, WINDOW - len(prompt), greedy, default_rng(0))
    b = generate(*uncached(network), prompt, WINDOW - len(prompt), greedy, default_rng(0))
    assert a == b


def test_one_model_and_one_seed_sample_the_same_text() -> None:
    network = _network()
    sampler = Sampler(temperature=0.9, top_k=4)
    first = generate(*cached(network), [3], 20, sampler, default_rng(7))
    assert generate(*cached(network), [3], 20, sampler, default_rng(7)) == first


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_train_then_sample_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str], backend: str) -> None:
    out = tmp_path / "model.json"
    argv = ["train", "--corpus", "euclid-heath", "--epochs", "1", "--seed", "0", "--limit", "8"]
    sample_text.main([*argv, "--out", str(out), "--backend", backend])
    assert "held-out" in capsys.readouterr().out
    network = load_network(str(out))
    assert network.implementation == backend
    tokenizer = CharacterTokenizer.load(tokenizer_path(out))
    assert tokenizer.corpus == "euclid-heath" and len(tokenizer) == network.class_count
    sample = ["sample", "--model", str(out), "--prompt", "Let", "--length", "40", "--seed", "3"]
    sample_text.main([*sample, "--temperature", "0.8", "--top-p", "0.9"])
    text = capsys.readouterr().out.rstrip("\n")
    assert text.startswith("Let") and len(text) == 43
    sample_text.main([*sample, "--temperature", "0.8", "--top-p", "0.9"])
    assert capsys.readouterr().out.rstrip("\n") == text


def test_dropout_builds_a_rotary_model_with_dropouts() -> None:
    specs = sample_text.model_specs(10, "rotary", 0.1)
    assert specs == sample_text.model_specs(10, "rotary", 0.1)
    assert "Position" not in repr(specs) and repr(specs).count("Dropout(p=0.1)") == 4
