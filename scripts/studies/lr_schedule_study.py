"""
The training-recipe study (the training-recipe workplan, stage 7, D7, D8, D11): the 2-layer rotary
transformer on every corpus and both held-out splits, 30 epochs, 5 seeds, under a constant rate,
warmup then cosine to a tenth, and the full recipe (that schedule, AdamW 0.1, clipping at 1.0).
numpy, under indrajala_ml/studies/lr_schedule_study.py's protocol and registered claims.

    python scripts/studies/lr_schedule_study.py time --corpus herodotus-rawlinson --arm recipe
    python scripts/studies/lr_schedule_study.py sweep --corpora herodotus-rawlinson muqaddimah --workers 8
    python scripts/studies/lr_schedule_study.py report --results A.json B.json

Findings: pending (the sweep runs on jebel and stone).
"""

from indrajala_ml.studies import lr_schedule_study

if __name__ == "__main__":
    lr_schedule_study.main(description=__doc__)
