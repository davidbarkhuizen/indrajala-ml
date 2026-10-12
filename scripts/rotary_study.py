"""
The rotary-positions study (the rotary-positions workplan, stage 8, D8): the sequence study's 1-layer
and 2-layer transformers with rotary attention and no Position, against the learned-position arms'
recorded results, on all four corpora; and the rotation's cost per epoch. numpy, under
indrajala_ml/studies/rotary_study.py's protocol.

    python scripts/rotary_study.py time --corpus herodotus-rawlinson --arm 2-layer
    python scripts/rotary_study.py sweep --control data/studies/attention-dropout/sweep-2026-10-10.json

Adam at batch 32, each corpus and arm at the sequence study's tuned rate, 10 epochs, 5 seeds,
OPENBLAS_NUM_THREADS=1, 6 workers.

Findings: not yet run.
"""

from indrajala_ml.studies import rotary_study

if __name__ == "__main__":
    rotary_study.main(description=__doc__)
