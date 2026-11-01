"""
The context-length study (the context-length workplan, stage 9, D8): the 2-layer rotary model under
the recipe at windows of 64, 256 and 1,024 characters, at 256 and 1,024 with and without the
document mask, on every corpus and both held-out splits, 5 seeds, at the token budget (B * T =
2,048), the training windows cut anew every epoch. Under
indrajala_ml/studies/context_length_study.py's protocol and the workplan's registered claims.

    python scripts/studies/context_length_study.py time --arm 1024 --backend rust --epochs E
    python scripts/studies/context_length_study.py sweep --corpora herodotus-rawlinson --epochs E --workers 8
    python scripts/studies/context_length_study.py report --results A.json B.json --epochs E

Findings: none yet (the study hasn't run).
"""

from indrajala_ml.studies import context_length_study

if __name__ == "__main__":
    context_length_study.main(description=__doc__)
