"""The sequence task's corpus and its windows (indrajala_ml/data/text_data.py)."""

from pathlib import Path

import pytest

from indrajala_ml.data.text_data import CONTEXT, TEXT_PATH, Vocabulary, load_text, load_text_dataset, split, windows


def test_a_vocabulary_is_the_texts_characters_sorted_and_round_trips():
    vocabulary = Vocabulary.of("hello, world\n")
    assert vocabulary.symbols == "\n ,dehlorw"
    assert len(vocabulary) == 10
    ids = vocabulary.encode("hello, world\n")
    assert ids[:5] == [5, 4, 6, 6, 7]
    assert vocabulary.decode(ids) == "hello, world\n"


def test_a_vocabularys_symbols_must_be_distinct_and_sorted():
    for symbols in ("ba", "aa"):
        with pytest.raises(AssertionError):
            Vocabulary(symbols)


def test_a_window_is_its_first_context_ids_as_floats_and_the_next_id_at_each_position():
    # two windows of 4 from 11 ids, the last 1 dropped
    assert windows(list(range(11)), context=3) == [
        ((0.0, 1.0, 2.0), (1, 2, 3)),
        ((4.0, 5.0, 6.0), (5, 6, 7)),
    ]
    assert windows([0, 1, 2], context=3) == []


def test_the_split_is_the_first_fraction_and_the_rest():
    assert split("abcdefghij", 0.9) == ("abcdefghi", "j")
    assert split("abcdefghij", 0.5) == ("abcde", "fghij")


def test_a_dataset_from_a_small_file(tmp_path: Path):
    path = tmp_path / "text.txt"
    path.write_text("abcabcabcabcabcabcabcabcabcabc")  # 30 characters
    train, held_out, vocabulary = load_text_dataset(str(path), context=2, train_fraction=0.8)

    assert vocabulary.symbols == "abc"
    # 24 training characters, 8 windows of 3; 6 held out, 2 windows
    assert len(train) == 8 and len(held_out) == 2
    assert train[0] == ((0.0, 1.0), (1, 2))
    assert held_out[0] == ((0.0, 1.0), (1, 2))  # held out starts at character 24, an "a"


# ---- the real corpus, fetched by ./cli setup (scripts/fetch_datasets.py)


def test_the_corpus_is_tiny_shakespeare():
    text = load_text(TEXT_PATH)
    assert len(text) == 1_115_394
    assert text.startswith("First Citizen:\nBefore we proceed any further, hear me speak.")
    assert Vocabulary.of(text).symbols == "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def test_the_corpus_gives_the_workplans_windows():
    train, held_out, vocabulary = load_text_dataset()

    assert len(vocabulary) == 65
    # the first 1,003,854 characters in windows of 65, and the last 111,540
    assert (len(train), len(held_out)) == (15_443, 1_716)
    state, labels = train[0]
    assert len(state) == len(labels) == CONTEXT
    assert vocabulary.decode([int(i) for i in state]) == load_text()[:CONTEXT]
    assert vocabulary.decode(labels) == load_text()[1 : CONTEXT + 1]
