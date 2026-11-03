"""
chrF (Popović 2015, "chrF: character n-gram F-score for automatic MT evaluation"; the
cross-attention workplan, D6): the character n-gram F-score production MT reports beside BLEU, as
sacreBLEU 2.x computes it by default (CHRF(char_order=6, word_order=0, beta=2)), so a score here is
comparable with the literature's.

Each line's whitespace is removed, then its character n-grams counted for n = 1 to 6. Over the
corpus, per order, the hypotheses' n-grams (none for a pair whose reference has no n-gram of that
order), the references' and their matches (each n-gram's smaller count) are summed. Per order with both counts positive, precision = matches / hypothesis
n-grams and recall = matches / reference n-grams; their means over those orders give
F_beta = (1 + beta^2) P R / (beta^2 P + R), times 100. Recall weighs beta^2 = 4 times precision.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

ORDER = 6
BETA = 2.0


def _ngrams(line: str, n: int) -> Counter[str]:
    stripped = "".join(line.split())
    return Counter(stripped[i : i + n] for i in range(len(stripped) - n + 1))


def statistics(hypothesis: str, reference: str, order: int = ORDER) -> list[int]:
    """One pair's [hypothesis n-grams, reference n-grams, matches] for n = 1 to order, flattened."""
    out: list[int] = []
    for n in range(1, order + 1):
        hyp, ref = _ngrams(hypothesis, n), _ngrams(reference, n)
        # a reference without n-grams of this order counts none of the hypothesis's (sacreBLEU's)
        out += [sum(hyp.values()) if ref else 0, sum(ref.values()), sum((hyp & ref).values())]
    return out


def score(totals: Sequence[int], order: int = ORDER, beta: float = BETA) -> float:
    """chrF from summed statistics: precision and recall averaged over the orders with both counts."""
    factor = beta**2
    precision = recall = 0.0
    effective = 0
    for i in range(order):
        n_hyp, n_ref, n_match = totals[3 * i : 3 * i + 3]
        if n_hyp > 0 and n_ref > 0:
            precision += n_match / n_hyp
            recall += n_match / n_ref
            effective += 1
    if effective == 0:
        return 0.0
    precision, recall = precision / effective, recall / effective
    if precision + recall == 0.0:
        return 0.0
    return 100 * (1 + factor) * precision * recall / (factor * precision + recall)


def chrf(hypotheses: Sequence[str], references: Sequence[str], order: int = ORDER, beta: float = BETA) -> float:
    """The corpus's chrF, one reference per hypothesis: every pair's statistics summed, then scored."""
    assert len(hypotheses) == len(references), (
        f"one reference per hypothesis; got {len(hypotheses)} and {len(references)}"
    )
    totals = [0] * (3 * order)
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        totals = [a + b for a, b in zip(totals, statistics(hypothesis, reference, order), strict=True)]
    return score(totals, order, beta)
