"""
The grouped-query attention study (the grouped-query attention workplan, stage 8, D7, D13): the
SwiGLU 256 model with 8 query heads sharing 8, 4, 2 and 1 key/value heads, trained from scratch,
and its multi-head runs uptrained to 4, 2 and 1, on the long-form corpora and both held-out splits,
5 seeds, on Rust under the recipe. Under indrajala_ml/studies/gqa_study.py's protocol and the
workplan's registered claims.

    python scripts/studies/gqa_study.py time --arm g8 --epochs 30
    python scripts/studies/gqa_study.py sweep --epochs 30 --workers 8 --corpora C...
    python scripts/studies/gqa_study.py uptrain --epochs 30 --workers 8 --corpora C...
    python scripts/studies/gqa_study.py report --results A.json ... --epochs 30

Findings: none yet (registered, not run).
"""

from indrajala_ml.studies import gqa_study

if __name__ == "__main__":
    gqa_study.main(description=__doc__)
