"""
The context-length study (the context-length workplan, stage 9, D8): the 2-layer rotary model under
the recipe at windows of 64, 256 and 1,024 characters, at 256 and 1,024 with and without the
document mask, on every corpus and both held-out splits, 5 seeds, at the token budget (B * T =
2,048), the training windows cut anew every epoch. Under
indrajala_ml/studies/context_length_study.py's protocol and the workplan's registered claims.

    python scripts/studies/context_length_study.py time --arm 1024 --backend rust --epochs 30
    python scripts/studies/context_length_study.py sweep --corpora herodotus-rawlinson muqaddimah --backend rust --epochs 30 --workers 8
    python scripts/studies/context_length_study.py report --results A.json B.json --epochs 30

Findings, 2026-11-01 to 11-02: jebel (i7-9700K) ran Herodotus and the Muqaddimah, stone (i7-1255U)
Euclid and Tiny Shakespeare, 8 workers each, on Rust, 30 epochs; their runs ended between 10:37
Sunday and 09:35 Monday (jebel) and 09:56 Sunday and 07:51 Monday (stone). All 200 runs completed.
Final-epoch held-out bits per character, means of 5 seeds; the report's tables, the result files and
the runs are in pyramidon's gitignored data/studies/context-length/ (report.md; manifest.tsv, every
file's path, host and sha256: jebel's sweep-herodotus-rawlinson-muqaddimah.json b9dbf3ee...,
stone's sweep-euclid-heath-tinyshakespeare.json 99097142...).

- C1, 1,024 below 64 (one-sided): holds. 26 of 30 pairs lower, mean -0.037 bits, Holm p 0.00012.
- C2, 256 below 64 (one-sided): holds. 30 of 30 pairs lower, mean -0.041 bits, Holm p 4.7e-09;
  -0.019 to -0.085 bits a cell, the most on Euclid's spread split (0.951 to 0.866).
- C3, 1,024 below 256 (one-sided): refuted. 12 of 30 pairs lower, mean +0.004 bits, Holm p 0.9.
  1,024 is lower on two cells (Herodotus's and Euclid's spread splits, -0.006 and -0.029) and
  higher on four (the Muqaddimah's two, +0.012 and +0.026; Herodotus's and Euclid's contiguous,
  +0.003 and +0.014). As registered, refuted means the step past 256 isn't shown to help at this
  model (2 layers, d = 64) and budget, not that it can't.
- C4, masked against unmasked at 256 (two-sided): undecided. 20 of 30 pairs lower with the mask,
  mean -0.004 bits, Holm p 0.2.
- C5, masked against unmasked at 1,024 (two-sided): differs, the mask lower. 24 of 30 pairs, mean
  -0.018 bits, Holm p 0.0043; the unmasked arm higher or equal in all six long-form cells (+0.000
  to +0.034), and slower in five of them (an epoch 266 to 402 s against 228 to 301: masked blocks
  are skipped; the Muqaddimah's spread split, 207 +- 37 s, the exception).
- By position in the window: at 1,024 the first sixteenth predicts far better than at 64 or 256
  (Euclid's spread split 0.834 bits, against 2.096 and 1.313), but the last sixteenth predicts
  worse than at 256 in five of the six long-form cells (Euclid's spread split 0.914 against 0.829;
  Herodotus's contiguous split the exception, 1.694 against 1.723). The model uses the longer
  window's early context, not its distant context, at this size; at the token budget the 1,024 arm
  also trains on 2 windows a step against 8.
- Convergence: held out still falls over the last three epochs in every cell (+0.001 to +0.012
  bits) except Euclid's contiguous split, which rises in every arm (-0.009 to -0.014): its training
  bits reach 0.650 to 0.772 against 1.336 to 1.369 held out, the shift between the book's parts.
- Cost: an epoch at 1,024 takes 1.6 to 2.8 times one at 64 (Herodotus 300 against 164 to 179 s;
  Euclid 236 to 243 against 87 to 100), at 256 1.2 to 1.5 times, so 1,024 costs 1.4 to 1.8 times
  256.
- Tiny Shakespeare (the reference, outside the claims) agrees: 256 below 64 on both splits (-0.031,
  -0.045), 1,024 level with 256 (-0.007, -0.005), the unmasked 1,024 arm level with 64 or above it
  (2.299 and 2.061 against 2.300 and 2.055).

So, as registered: later sequence work on the long-form corpora uses windows of 256 (C2 holds; C3
says 1,024 doesn't earn its 1.4-1.8x cost over 256 at this model), and the document mask stays the
packed default (C5; C4 undecided).
"""

from indrajala_ml.studies import context_length_study

if __name__ == "__main__":
    context_length_study.main(description=__doc__)
