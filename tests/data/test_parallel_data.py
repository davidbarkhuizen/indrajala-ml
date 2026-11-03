"""Parallel pairs packed into windows (the cross-attention workplan, stage 2, D11, D13)."""

import json
import os
from pathlib import Path
from random import Random

import numpy as np
import pytest

from indrajala_ml.data.parallel_data import (
    PARALLEL_CORPORA,
    Pair,
    fits,
    load_pairs,
    load_parallel_dataset,
    pack_pairs,
)
from indrajala_ml.data.text_data import Vocabulary, pad_id, separator_id
from indrajala_ml.model.layers.numpy.segments import segment_context

SOURCE = Vocabulary.of("αβγ")
TARGET = Vocabulary.of("abc")


def pair(grc: str, eng: str, split: str = "train", ref: str = "1") -> Pair:
    return Pair("work", ref, split, grc, eng)  # type: ignore[arg-type]


def write(path: Path, pairs: list[Pair]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps({"work": p.work, "ref": p.ref, "split": p.split, "grc": p.grc, "eng": p.eng}) + "\n"
            for p in pairs
        )
    return str(path)


def test_a_window_is_the_greek_each_followed_by_the_separator_and_the_target_row_opened_by_one():
    S, P = separator_id(SOURCE), pad_id(SOURCE)
    s, p = separator_id(TARGET), pad_id(TARGET)
    [window] = pack_pairs([pair("αβ", "ab"), pair("γ", "c")], SOURCE, TARGET, window=8)
    assert window.pairs == 2
    # the source: αβ S γ S, left-padded to 8
    assert window.source == (P, P, P, 0, 1, S, 2, S)
    # the target row: s ab s c s, left-padded to 9; the decoder reads all but the last, predicts all but the first
    assert window.target == (p, p, p, s, 0, 1, s, 2, s)
    assert window.decoder_input == window.target[:-1] and window.labels == window.target[1:]


def test_packing_is_greedy_and_each_side_can_close_a_window():
    # the source closes the first window: 3 + 3 > 5
    windows = pack_pairs([pair("αα", "a"), pair("ββ", "b"), pair("γ", "c")], SOURCE, TARGET, window=5)
    assert [w.pairs for w in windows] == [1, 2]
    # the target closes it: the rows' English, separators counted, 3 + 3 > 5 while the Greek fits
    windows = pack_pairs([pair("α", "aa"), pair("β", "bb"), pair("γ", "c")], SOURCE, TARGET, window=5)
    assert [w.pairs for w in windows] == [1, 2]
    for w in windows:
        assert len(w.source) == 5 and len(w.target) == 6


def test_pair_k_is_segment_k_on_both_sides():
    pairs = [pair("αβ", "ab"), pair("γγγ", "c"), pair("β", "bca")]
    [window] = pack_pairs(pairs, SOURCE, TARGET, window=12)
    source = np.array([window.source], dtype=np.float64)
    # the source by step 11's rule: a separator ends its segment
    source_segments = segment_context(source, separator_id(SOURCE), pad_id(SOURCE)).segments[0]
    # the decoder's input by D13's: a separator opens the segment after it
    decoder_input = np.array(window.decoder_input)
    target_segments = np.where(decoder_input == pad_id(TARGET), 0, np.cumsum(decoder_input == separator_id(TARGET)))
    for k, p in enumerate(pairs, start=1):
        greek = [
            SOURCE.symbols[i] for i, g in zip(window.source, source_segments, strict=True) if g == k and i < len(SOURCE)
        ]
        english_read = [
            TARGET.symbols[i] for i, g in zip(decoder_input, target_segments, strict=True) if g == k and i < len(TARGET)
        ]
        english_predicted = [
            TARGET.symbols[i] for i, g in zip(window.labels, target_segments, strict=True) if g == k and i < len(TARGET)
        ]
        assert "".join(greek) == p.grc
        assert "".join(english_predicted) == p.eng  # every character, the first included, from its own pair
        assert "".join(english_read) == p.eng  # read after the separator that opens the pair
    # each pair's last label is the separator that ends its translation (D7), in its own segment
    # (the opening separator's label sits at a pad's position, segment 0: given, not predicted)
    ends = [g for i, g in zip(window.labels, target_segments, strict=True) if i == separator_id(TARGET) and g]
    assert ends == [1, 2, 3]
    # the opening separator is given: its target position's input is a pad, so it doesn't count (step 11's D6)
    counted = segment_context(
        np.array([window.decoder_input], dtype=np.float64), separator_id(TARGET), pad_id(TARGET)
    ).counted[0]
    first = window.target.index(separator_id(TARGET))
    assert counted[first - 1] == 0.0 and counted[first] == 1.0


def test_a_pair_longer_than_a_window_is_dropped_never_cut(tmp_path: Path):
    pairs = [
        pair("α" * 7, "a", "train", "1"),  # source 8: fits 8
        pair("α" * 8, "a", "train", "2"),  # source 9: doesn't
        pair("β", "b" * 8, "validation", "3"),  # target row 1 + 9 > 9: doesn't
        pair("γ", "c" * 7, "test", "4"),  # target row 1 + 8 = 9: fits
    ]
    assert [fits(p, 8) for p in pairs] == [True, False, False, True]
    dataset = load_parallel_dataset(window=8, path=write(tmp_path / "pairs.jsonl", pairs))
    assert [p.ref for p in dataset.dropped] == ["2", "3"]
    assert [p.ref for p in dataset.train.pairs] == ["1"]
    assert not dataset.validation and [w.pairs for w in dataset.test] == [1]
    with pytest.raises(AssertionError):
        pack_pairs([pairs[1]], SOURCE, TARGET, window=8)


def test_the_vocabularies_are_each_side_s_characters_over_every_split(tmp_path: Path):
    pairs = [pair("αβ", "ab", "train"), pair("γ", "c", "test")]
    dataset = load_parallel_dataset(window=8, path=write(tmp_path / "pairs.jsonl", pairs))
    assert dataset.source.symbols == "αβγ" and dataset.target.symbols == "abc"


def test_load_pairs_reads_text_order_and_refuses_an_unknown_split(tmp_path: Path):
    pairs = [pair("α", "a", "train", "1.1"), pair("β", "b", "validation", "1.2")]
    assert load_pairs(write(tmp_path / "ok.jsonl", pairs)) == pairs
    with pytest.raises(AssertionError, match="a split is one of"):
        load_pairs(write(tmp_path / "bad.jsonl", [pair("α", "a", "dev")]))


def test_training_windows_are_packed_anew_each_epoch_reproducibly(tmp_path: Path):
    pairs = [pair("α" * (1 + i % 3), "a" * (1 + i % 2), "train", str(i)) for i in range(40)]
    dataset = load_parallel_dataset(window=8, path=write(tmp_path / "pairs.jsonl", pairs))
    first, again = dataset.train.epoch(Random(7)), dataset.train.epoch(Random(7))
    assert first == again
    assert first != dataset.train.epoch(Random(8))
    assert sum(w.pairs for w in first) == len(pairs)  # every pair once an epoch


@pytest.mark.skipif(
    not all(os.path.exists(path) for path in PARALLEL_CORPORA.values()), reason="the parallel corpora aren't fetched"
)
@pytest.mark.parametrize("corpus", list(PARALLEL_CORPORA))
def test_the_corpora_pack_at_1024(corpus: str):
    dataset = load_parallel_dataset(corpus)
    pairs = len(dataset.train.pairs) + sum(w.pairs for w in dataset.validation + dataset.test) + len(dataset.dropped)
    assert pairs == {"histories": 14_602, "new-testament": 7_908}[corpus]
    # the alignment report's 68 pairs over 1,024 (a side longer than the window), and 2 of exactly 1,024,
    # whose separator overflows it
    assert len(dataset.dropped) == {"histories": 70, "new-testament": 0}[corpus]
    assert all(max(len(p.grc), len(p.eng)) >= 1024 for p in dataset.dropped)
    for window in dataset.validation[:5] + dataset.train.epoch(Random(0))[:5]:
        assert len(window.source) == 1024 and len(window.target) == 1025
