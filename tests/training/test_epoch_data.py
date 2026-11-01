"""
Training windows cut anew every epoch (the context-length workplan, D4) and the token budget (D5):
PackedEpochs packs the training paragraphs in an order the trainer's generator draws, so every
paragraph is seen once an epoch and the windows' count stays while their cuts change; the trainer
draws an epoch's windows at its start, so a resumed run cuts and trains as the run in one go, by
bits; and a list trains as before.
"""

from __future__ import annotations

from random import Random
from typing import Any

import pytest

from indrajala_ml.data import text_data
from indrajala_ml.data.text_data import PackedEpochs, Vocabulary, budget_batch_size, pack, paragraphs
from indrajala_ml.training.train import EpochData, train_backprop_network_mini_batch
from tests.helpers import bits
from tests.model.networks.test_segments_numpy import T, _network

# paragraphs over a 5-symbol vocabulary (the segments tests' network's), of 1 to 19 characters
TEXT = "\n\n".join("abcde"[k % 5] * (1 + (7 * k) % 19) + "cab"[: k % 3] for k in range(40))
VOCABULARY = Vocabulary.of("abcde")  # the separator is id 5, the network's


def _epochs(context: int = T) -> PackedEpochs:
    return PackedEpochs(tuple(paragraphs(TEXT)), VOCABULARY, context)


def test_an_epoch_packs_every_paragraph_once_in_the_order_the_generator_draws():
    epochs = _epochs()
    rng = Random(3)
    order = list(epochs.pieces)
    Random(3).shuffle(order)
    assert epochs.epoch(rng) == text_data.windows(pack(order, VOCABULARY), T)


def test_every_epoch_has_the_same_count_of_windows_and_other_cuts():
    epochs = _epochs()
    rng = Random(5)
    drawn = [epochs.epoch(rng) for _ in range(4)]
    assert {len(windows) for windows in drawn} == {len(epochs)}
    assert len(epochs) == sum(len(piece) + 1 for piece in epochs.pieces) // (T + 1)
    assert len({tuple(windows) for windows in drawn}) == 4
    assert isinstance(epochs, EpochData)


def test_the_held_out_windows_are_the_packed_loaders_and_the_same_paragraphs_train(tmp_path: Any):
    path = tmp_path / "text.txt"
    path.write_text(TEXT)
    pieces = paragraphs(TEXT)
    for split_by in ("contiguous", "spread"):
        epochs, held_out, vocabulary = text_data.load_packed_text_epochs(str(path), T, split_by=split_by)  # pyright: ignore[reportArgumentType]
        _train, expected, expected_vocabulary = text_data.load_packed_text_dataset(str(path), T, split_by=split_by)  # pyright: ignore[reportArgumentType]
        assert held_out == expected and vocabulary == expected_vocabulary
        if split_by == "contiguous":
            trained = text_data.split_paragraphs(pieces)[0]
        else:
            trained = [piece for block in text_data.spread_split_paragraphs(pieces)[0] for piece in block]
        assert list(epochs.pieces) == trained


def _train(network: Any, data: Any, epochs: int, rng: Random, resume_from: Any = None) -> Any:
    return train_backprop_network_mini_batch(
        network, data, batch_size=4, learning_rate=0.05, epochs=epochs, rng=rng, resume_from=resume_from
    )


def test_a_resumed_run_cuts_and_trains_as_the_run_in_one_go_by_bits():
    in_one_go = _network("rotary")
    _train(in_one_go, _epochs(), 4, Random(7))

    stopped = _network("rotary")
    run = _train(stopped, _epochs(), 2, Random(7)).run_checkpoint
    assert run is not None
    finished = _network("rotary", seed=99)  # other weights: the checkpoint replaces them
    result = _train(finished, _epochs(), 4, Random(), resume_from=run)
    assert result.run_checkpoint is not None
    assert bits(finished.snapshot()) == bits(in_one_go.snapshot())


def test_a_list_trains_as_before_and_an_epoch_data_moves_on():
    fixed = _epochs().epoch(Random(1))
    a, b = _network("rotary"), _network("rotary")
    _train(a, fixed, 2, Random(7))
    _train(b, list(fixed), 2, Random(7))
    assert bits(a.snapshot()) == bits(b.snapshot())
    c = _network("rotary")
    _train(c, _epochs(), 2, Random(7))
    assert bits(c.snapshot()) != bits(a.snapshot())


@pytest.mark.parametrize(("context", "batch"), [(64, 32), (256, 8), (1024, 2), (2048, 1)])
def test_the_token_budget_keeps_every_steps_tokens(context: int, batch: int):
    assert budget_batch_size(context) == batch and batch * context == text_data.TOKEN_BUDGET


def test_a_context_that_doesnt_divide_the_budget_is_refused():
    with pytest.raises(AssertionError, match="divides the token budget"):
        budget_batch_size(100)
