"""
The attention-dropout study (the attention-dropout workplan, stage 8, D8): does dropout close the
sequence study's train/held-out gaps on Euclid (Heath), Tiny Shakespeare and Herodotus
(Rawlinson), and is the gap overfitting or the held-out text's shift? numpy, under
indrajala_ml/studies/attention_dropout_study.py's protocol.

    python scripts/attention_dropout_study.py time --corpus herodotus-rawlinson --arm "both 0.2"
    python scripts/attention_dropout_study.py sweep --epochs 10 --seeds 5 --out sweep.json

Findings: to come with the sweep.
"""

from indrajala_ml.studies import attention_dropout_study

if __name__ == "__main__":
    attention_dropout_study.main(description=__doc__)
