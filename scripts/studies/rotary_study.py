"""
The rotary-positions study (the rotary-positions workplan, stage 8, D8): the sequence study's 1-layer
and 2-layer transformers with rotary attention and no Position, against the learned-position arms'
recorded results, on all four corpora; and the rotation's cost per epoch. numpy, under
indrajala_ml/studies/rotary_study.py's protocol.

    python scripts/studies/rotary_study.py time --corpus herodotus-rawlinson --arm 2-layer
    python scripts/studies/rotary_study.py sweep --control data/studies/attention-dropout/sweep-2026-10-10.json

Adam at batch 32, each corpus and arm at the sequence study's tuned rate, 10 epochs, 5 seeds,
OPENBLAS_NUM_THREADS=1, 6 workers.

Findings, on jebel (i7-9700K), nothing else running: the sweep from 14:24 to 16:47 on 2026-10-12
(2 h 23 min), its results `data/studies/rotary/sweep-2026-10-12-1647.json` on jebel (sha256
2a987763dcd1d70ad018a63a504e2f0f109efd70ec11ea2f46d972c9c762bddf; commit d9b14b6), and `time`'s
`data/studies/rotary/time-2026-10-12-1423.json` (sha256
dd438c4e9707b7b40490924bc1b29bb80fee63bd8817a15e8d76063b6901e79b). Held-out bits per character at
epoch 10, rotary minus learned per seed, mean ± sd over 5 seeds.

- The run is valid: every learned row, the 25 rerun and the three 2-layer rows taken from the
  attention-dropout control, reproduced the recorded means at epochs 1, 5 and 10 (the sweep exits
  non-zero otherwise; it exited 0).
- Rotary is lower held out on every corpus and arm, on 39 of 40 seeds: Tiny Shakespeare -0.041 ±
  0.015 (1-layer) and -0.051 ± 0.009 (2-layer), Herodotus -0.045 ± 0.006 and -0.035 ± 0.011, the
  *Muqaddimah* -0.024 ± 0.022 and -0.035 ± 0.016, Euclid -0.055 ± 0.030 and -0.062 ± 0.008. The
  one exception is a *Muqaddimah* 1-layer seed, +0.003. Accuracy rises 0.65 to 0.89 points, with
  4,096 fewer parameters (no 64 × 64 Position table).
- The lead is largest early: -0.081 to -0.194 bits at epoch 1, narrowing by epoch 10 but never
  closing. Rotary lowers the training loss alike, so the gaps between held-out and training bits
  move by under 0.02 either way, but for Euclid 2-layer's 0.051 (0.438 against 0.489): it fits
  faster; it mostly doesn't overfit less.
- The rotation's cost: alone (`time`), a 2-layer Herodotus epoch took 56.1 s against 54.2 s
  learned (+3.5%). In the sweep at 6 workers the rows run side by side cost 7.2 to 9.6% more (the
  1-layer arms, and the *Muqaddimah*'s 2-layer). The three 2-layer learned rows from the control
  ran on 2026-10-10 in another sweep, so their seconds don't compare.
- D9 (a): rotary at least matches learned positions everywhere, so new sequence models default to
  rotary attention without a Position (the README's example, and step 9's `train`).
"""

from indrajala_ml.studies import rotary_study

if __name__ == "__main__":
    rotary_study.main(description=__doc__)
