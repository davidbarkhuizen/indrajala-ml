"""
chrF (the cross-attention workplan, D6): sacreBLEU 2.5.1's default CHRF() corpus scores, recorded in
tests/fixtures/chrf/sacrebleu-2.5.1.json from random English and Greek pairs (empty lines and lines
shorter than an order among them), reproduced to rounding; and by hand: identical lines score 100,
whitespace is ignored, and a line sharing nothing scores 0.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.generation.chrf import chrf, statistics

CASES = json.loads((Path(__file__).parents[1] / "fixtures" / "chrf" / "sacrebleu-2.5.1.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[str(i) for i in range(len(CASES))])
def test_the_scores_are_sacrebleus(case: dict[str, Any]):
    hypotheses: list[str] = case["hypotheses"]
    references: list[str] = case["references"]
    assert chrf(hypotheses, references) == pytest.approx(case["score"], rel=1e-12, abs=1e-12)


def test_by_hand():
    assert chrf(["the ships"], ["the ships"]) == pytest.approx(100.0)
    assert chrf(["theships"], ["the  ships "]) == pytest.approx(100.0)
    assert chrf(["xyz"], ["abc"]) == 0.0
    # "ab" against "abc": unigrams 2/2 hyp, 2/3 ref; bigrams 1/1, 1/2; no trigram in the hypothesis
    assert statistics("ab", "abc", 3) == [2, 3, 2, 1, 2, 1, 0, 1, 0]
