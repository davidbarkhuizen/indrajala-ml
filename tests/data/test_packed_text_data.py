"""Packed windows of paragraphs and the special tokens (the segments workplan, stage 3, D2, D7)."""

from pathlib import Path

import pytest

from indrajala_ml.data import text_data
from indrajala_ml.data.text_data import (
    PAD,
    SEPARATOR,
    SPECIAL_TOKENS,
    Vocabulary,
    load_packed_text_dataset,
    pack,
    pad_id,
    paragraphs,
    separator_id,
    split_paragraphs,
    spread_split_paragraphs,
)
from indrajala_ml.generation.tokenizer import CharacterTokenizer


def test_paragraphs_are_the_pieces_between_runs_of_blank_lines():
    text = "\n\nFirst line\nsecond line\n\n\nNext.\n  \t\nLast\n\n"
    assert paragraphs(text) == ["First line\nsecond line", "Next.", "Last"]
    assert paragraphs("one paragraph\nof two lines") == ["one paragraph\nof two lines"]
    assert paragraphs("\n\n \n") == []


def test_packing_follows_each_paragraph_with_the_separator():
    vocabulary = Vocabulary.of("ab c")
    assert separator_id(vocabulary) == 4 and pad_id(vocabulary) == 5
    assert pack(["ab", "c"], vocabulary) == [1, 2, 4, 3, 4]


def test_the_contiguous_split_cuts_at_the_paragraph_boundary_nearest_the_fraction():
    pieces = ["a" * 40, "b" * 40, "c" * 9, "d" * 9]  # boundaries at 41, 82, 92, 102 of 102 characters
    assert split_paragraphs(pieces, 0.9) == (pieces[:3], pieces[3:])  # 91.8: 92
    assert split_paragraphs(pieces, 0.5) == (pieces[:1], pieces[1:])  # 51: 41
    assert split_paragraphs(pieces, 0.62) == (pieces[:2], pieces[2:])  # 63.24: 82 over 41
    assert split_paragraphs(["only"], 0.9) == (["only"], [])


def test_the_spread_split_keeps_whole_paragraphs_in_blocks_and_carries_the_rounding():
    pieces = [f"{i:02d}" * 5 for i in range(40)]  # 40 paragraphs of 10 characters
    train, held_out = spread_split_paragraphs(pieces, 0.75, blocks=4)
    # 7.5 paragraphs a block: a tie trains 8, and the half paragraph carried makes the next 7
    assert [len(block) for block in train] == [8, 7, 8, 7] and [len(block) for block in held_out] == [2, 3, 2, 3]
    seen = [piece for block in train + held_out for piece in block]
    assert sorted(seen) == sorted(pieces)
    assert train[1][0] == pieces[10] and held_out[1] == pieces[17:20]


Window = tuple[tuple[float, ...], tuple[int, ...]]


def _numbers(windows: list[Window], vocabulary: Vocabulary) -> set[str]:
    # the paragraph numbers a split's windows hold, read back through the separators
    separator = separator_id(vocabulary)
    text = "".join(
        SEPARATOR if i == separator else vocabulary.symbols[i]
        for state, labels in windows
        for i in (int(state[0]), *labels)
    )
    return {piece.split()[1] for piece in text.split(SEPARATOR) if piece.startswith("paragraph ")}


@pytest.mark.parametrize("split_by", ["contiguous", "spread"])
def test_a_packed_dataset_holds_no_paragraph_on_both_sides(tmp_path: Path, split_by: text_data.Split):
    # 2,000 paragraphs: 20 to each of the spread split's 100 blocks
    path = tmp_path / "corpus.txt"
    text = "\n\n".join(f"paragraph {i} " + "xyz" * (i % 7 + 1) for i in range(2000)) + "\n"
    path.write_text(text, encoding="utf-8")
    train, held_out, vocabulary = load_packed_text_dataset(str(path), context=8, split_by=split_by)
    assert train and held_out
    for state, labels in train + held_out:
        ids = [int(x) for x in state]
        assert labels[:-1] == tuple(ids[1:])
        assert pad_id(vocabulary) not in ids and pad_id(vocabulary) not in labels
    assert any(separator_id(vocabulary) in labels for _state, labels in train)
    assert not _numbers(train, vocabulary) & _numbers(held_out, vocabulary)
    assert 0.08 < len(held_out) / (len(train) + len(held_out)) < 0.12


def test_the_fixed_window_loader_is_unchanged(tmp_path: Path):
    path = tmp_path / "corpus.txt"
    path.write_text("ab\n\ncd\n\nef\n" * 10, encoding="utf-8")
    train, _held_out, vocabulary = text_data.load_text_dataset(str(path), context=4)
    assert "\n" in vocabulary.symbols and all(max(labels) < len(vocabulary) for _state, labels in train)


def test_a_tokenizer_with_the_special_tokens_counts_them_after_the_symbols(tmp_path: Path):
    tokenizer = CharacterTokenizer.of("toy", "ab\n\nc", special=SPECIAL_TOKENS)
    assert len(tokenizer) == 6 and tokenizer.output_classes == 5
    assert tokenizer.special_id(SEPARATOR) == 4 and tokenizer.special_id(PAD) == 5
    assert tokenizer.decode([1, 4, 3]) == f"a{SEPARATOR}c"

    path = tmp_path / "toy.tokenizer.json"
    tokenizer.save(path)
    assert CharacterTokenizer.load(path) == tokenizer

    plain = CharacterTokenizer.of("toy", "ab\n\nc")
    plain.save(path)
    assert "special" not in path.read_text() and plain.output_classes == len(plain) == 4
    with pytest.raises(ValueError, match="no '<|sep|>'"):
        plain.special_id(SEPARATOR)
    with pytest.raises(ValueError, match="special tokens"):
        CharacterTokenizer("toy", "", "ab", ("<|bos|>",))


def test_the_spread_split_holds_out_its_fraction_with_few_paragraphs_a_block():
    # five long paragraphs a block, as the Muqaddimah's 540 over 100 blocks: rounding each block
    # alone held out 4% of 10%; carried, the split holds out its fraction
    pieces = [chr(65 + i % 26) * (900 + 37 * (i % 11)) for i in range(500)]
    _train, held_out = spread_split_paragraphs(pieces, 0.9)
    held = sum(len(piece) + 1 for block in held_out for piece in block)
    total = sum(len(piece) + 1 for piece in pieces)
    assert 0.095 < held / total < 0.105
    assert len(held_out) >= 40  # spread across the text, not one block
