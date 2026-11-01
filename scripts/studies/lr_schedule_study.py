"""
The training-recipe study (the training-recipe workplan, stage 7, D7, D8, D11): the 2-layer rotary
transformer on every corpus and both held-out splits, 30 epochs, 5 seeds, under a constant rate,
warmup then cosine to a tenth, and the full recipe (that schedule, AdamW 0.1, clipping at 1.0).
numpy, under indrajala_ml/studies/lr_schedule_study.py's protocol and registered claims.

    python scripts/studies/lr_schedule_study.py time --corpus herodotus-rawlinson --arm recipe
    python scripts/studies/lr_schedule_study.py sweep --corpora herodotus-rawlinson muqaddimah --workers 8
    python scripts/studies/lr_schedule_study.py report --results A.json B.json

Findings, 2026-10-31 to 11-01: jebel (i7-9700K) ran Herodotus and the Muqaddimah at 8 workers in
11.8 hours, stone (i7-1255U, 12 threads) Tiny Shakespeare and Euclid at 12 in 8.7, nothing else running; an
epoch took 136 to 217 s a job (each host's last runs, with fewer jobs beside them, ran faster:
the 88.0 and 130.5 s rows). All 120 runs stayed finite for 30 epochs. Final-epoch held-out bits per
character, means of 5 seeds; the report's tables are in pyramidon's gitignored
data/studies/lr-schedule/report.txt, beside the runs and seed 0's models.

- C1, cosine below constant (one-sided): holds. 36 of 40 pairs lower (26 needed), mean -0.029
  bits, Holm p 9.3e-08. Every seed of 6 of the 8 cells, -0.023 to -0.055 bits a cell; the
  Muqaddimah's contiguous split 4 of 5 (-0.009); Euclid's contiguous split 2 of 5, +0.015 bits
  higher on the mean. D8's caveat (a peak tuned for a constant rate) didn't decide C1; whether a
  higher peak helps the cosine more is untested.
- C2, the recipe against cosine (two-sided): differs, lower. 40 of 40 pairs lower, mean -0.021
  bits, Holm p 3.6e-12; -0.009 to -0.040 bits a cell, the most on Tiny Shakespeare's contiguous
  split and Euclid's (-0.040, -0.034) and the Muqaddimah's (-0.030). The recipe is cosine plus
  AdamW 0.1 and clipping at 1.0 together: the study doesn't say which of the two gives it.
- Where the cosine loses, it overfits into the shift: on Euclid's contiguous split its held-out
  loss rises over the last three epochs (-0.015 bits "fell") while its training bits fall to 0.790
  (constant 0.875); the recipe's weight decay brings it back below constant (1.420 against
  1.440). On Euclid's spread split, the same text without the shift, the cosine wins on every seed.
- The decay lowers the training loss more than the held-out one (training bits 0.064 to 0.100
  lower under cosine on every cell, held out at most 0.055, and higher on Euclid's contiguous
  split): the constant rate's noise had kept training from its minimum. At epoch 15 the cosine is
  already lower than constant on every cell, while the recipe is above the cosine on 6 of 8 (equal
  on Tiny Shakespeare's contiguous split, lower on Euclid's) and passes it by epoch 30 on all 8.
- Convergence: the constant arms still fall 0.000 to 0.009 bits over the last three epochs; the
  cosine and recipe arms, at their final, lowest rate, move -0.015 to +0.006.

So the default for new sequence models is the recipe: warmup then cosine to a tenth, AdamW 0.1,
clipping at 1.0 (the workplan's step 10).
"""

from indrajala_ml.studies import lr_schedule_study

if __name__ == "__main__":
    lr_schedule_study.main(description=__doc__)
