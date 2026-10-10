"""The sequence task's corpus and its windows (indrajala_ml/data/text_data.py)."""

from pathlib import Path

import pytest

from indrajala_ml.data.text_data import (
    CONTEXT,
    CORPORA,
    TEXT_PATH,
    Vocabulary,
    load_text,
    load_text_dataset,
    split,
    spread_split,
    windows,
)


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


def test_the_spread_split_holds_out_each_blocks_last_part():
    # 10 blocks of 10 characters (block 3 is "dddddddddd"), each its first 9 then its last
    text = "".join(chr(ord("a") + b) * 10 for b in range(10))
    train, held_out = spread_split(text, 0.9, blocks=10)
    assert train == [chr(ord("a") + b) * 9 for b in range(10)]
    assert held_out == [chr(ord("a") + b) for b in range(10)]
    # unequal blocks: b * n // blocks, so every character is in one piece, in order
    train, held_out = spread_split("abcdefghijk", 0.5, blocks=3)
    assert [t + h for t, h in zip(train, held_out)] == ["abc", "defg", "hijk"]
    assert (train, held_out) == (["a", "de", "hi"], ["bc", "fg", "jk"])


def test_a_spread_dataset_cuts_each_piece_into_its_own_windows(tmp_path: Path):
    path = tmp_path / "text.txt"
    # 100 blocks of 30 characters; each block's first 27 (9 windows of 3) train, its last 3 (1) held out
    path.write_text("abc" * 1000)
    train, held_out, vocabulary = load_text_dataset(str(path), context=2, split_by="spread")
    assert vocabulary.symbols == "abc"
    assert (len(train), len(held_out)) == (900, 100)
    # a held-out piece starts at character 27 of its block, an "a": no window crosses into it
    assert set(held_out) == {((0.0, 1.0), (1, 2))}
    with pytest.raises(AssertionError, match="contiguous or spread"):
        load_text_dataset(str(path), split_by="random")  # pyright: ignore[reportArgumentType]


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


@pytest.mark.parametrize(
    ("corpus", "length", "symbols", "windows_per_part"),
    [
        ("tinyshakespeare", 1_115_394, 65, (15_443, 1_716)),
        ("herodotus-rawlinson", 1_496_601, 76, (20_722, 2_302)),
        ("muqaddimah", 1_012_838, 40, (14_023, 1_558)),
        ("euclid-heath", 823_448, 67, (11_401, 1_266)),
    ],
)
def test_each_corpus_gives_its_windows(corpus: str, length: int, symbols: int, windows_per_part: tuple[int, int]):
    text = load_text(CORPORA[corpus])
    train, held_out, vocabulary = load_text_dataset(CORPORA[corpus])

    assert (len(text), len(vocabulary)) == (length, symbols)
    assert (len(train), len(held_out)) == windows_per_part
    assert vocabulary.decode(train[0][1]) == text[1 : CONTEXT + 1]


@pytest.mark.parametrize("corpus", list(CORPORA))
def test_each_corpus_spread_holds_out_about_its_contiguous_share(corpus: str):
    contiguous = load_text_dataset(CORPORA[corpus])
    spread = load_text_dataset(CORPORA[corpus], split_by="spread")
    # the 200 pieces' remainders are dropped: within 10% of the contiguous split's windows
    for part in (0, 1):
        assert 0.9 * len(contiguous[part]) <= len(spread[part]) <= len(contiguous[part])
    assert spread[2] == contiguous[2]


def test_the_muqaddimah_is_arabic_letters_space_newline_and_parentheses():
    vocabulary = Vocabulary.of(load_text(CORPORA["muqaddimah"]))
    assert vocabulary.symbols[:4] == "\n ()"
    assert all("\u0621" <= symbol <= "\u064a" for symbol in vocabulary.symbols[4:])
