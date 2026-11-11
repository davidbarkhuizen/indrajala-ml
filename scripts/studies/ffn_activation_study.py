"""
The FFN activations study (the FFN activations workplan, stage 6, D7): the context-length study's
256 model with its FFNs ReLU, GELU and SwiGLU at equal parameters, on every corpus and both
held-out splits, 5 seeds, on Rust under the recipe for 30 epochs. ReLU's runs are the
context-length study's (D11). Under indrajala_ml/studies/ffn_activation_study.py's protocol and the
workplan's registered claims.

    python scripts/studies/ffn_activation_study.py time --arm relu --epochs 30 --control CONTROL...
    python scripts/studies/ffn_activation_study.py time --arm swiglu --epochs 30
    python scripts/studies/ffn_activation_study.py sweep --epochs 30 --workers 8
    python scripts/studies/ffn_activation_study.py report --results A.json --control CONTROL... --epochs 30

Findings: none. The study was abandoned by the owner on 2026-11-03 after 24 of its 80 runs, on
stone at 8 workers far slower than its registered estimate; F1-F3 are untested and the partial runs
not analysed (next-steps.md, Sequence models and language). The code stays: the GQA study builds its model
from arm_specs("swiglu").
"""

from indrajala_ml.studies import ffn_activation_study

if __name__ == "__main__":
    ffn_activation_study.main(description=__doc__)
