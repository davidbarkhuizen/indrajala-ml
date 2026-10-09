"""
The sequence task's study (the sequence task workplan, stage 8, D9): next-character prediction on
Tiny Shakespeare, Herodotus (Rawlinson), the Muqaddimah and Euclid's Elements (Heath) (D2), numpy,
under indrajala_ml/studies/sequence_study.py's protocol.

    python scripts/sequence_study.py time --corpus herodotus-rawlinson --arm 2-layer
    python scripts/sequence_study.py tune --out tune.json
    python scripts/sequence_study.py sweep --tuned tune.json --epochs N --out sweep.json

The questions (D9): does attention over the past beat a per-token model (1-layer against ffn, and
both against the counted unigram and bigram floors); does depth help (2-layer against 1-layer);
and does the mask do its job (2-layer-unmasked, the leak, should fall far below every causal arm,
held out included). And across the corpora (D2), what differs between them; the workplan's naive
predictions about Euclid are tested here.

Findings: not yet run.
"""

from indrajala_ml.studies import sequence_study

if __name__ == "__main__":
    sequence_study.main(description=__doc__)
