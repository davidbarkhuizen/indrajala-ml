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

Adam at batch 32, one rate per corpus and arm from `tune`, 10 epochs, 5 seeds,
OPENBLAS_NUM_THREADS=1, 6 workers.

Findings, on jebel (i7-9700K), nothing else running. One epoch of the slowest cell, 2-layer on
Herodotus, took 54.4 s alone (`time`), so `tune` ran every corpus and arm at five rates (0.0005 to
0.008) for 2 epochs and 3 seeds, and `sweep` ran 10 epochs and 5 seeds at each one's best rate.
Every run stayed finite.

- tune: the attention arms' best rates were 0.002 or 0.004 on every corpus, never at either end of
  the range; ffn's were 0.0005 or 0.001, its rates within 0.02 bits of each other.
- A per-token model is a bigram model: ffn ends 0.008 to 0.011 bits per character above the
  counted bigram floor on every corpus, and no rate or epoch moves it. A token that sees only
  itself and its position can learn the next character's frequency after this one, and nothing
  more.
- Attention over the past is the gain: 1-layer is 0.80 (the Muqaddimah) to 1.52 (Euclid) bits per
  character under ffn, on every seed; per-token accuracy rises from 27-38% to 45-71%.
- Depth helps on every corpus and seed: a second layer takes 0.09 (the Muqaddimah) to 0.17
  (Herodotus) bits per character more, at 1.8x the time per epoch. Neither transformer had
  converged: their losses still fell by 0.01 to 0.05 over the last three epochs.
- The mask does its job, and the leak is the size D9 expected: 2-layer-unmasked reaches 0.044 to
  0.056 bits per character and 99% accuracy held out on every corpus, by the first epoch, 1.4 to
  2.7 bits under 2-layer: it copies the next input. Its loss is the same on every corpus; copying
  doesn't depend on the text.
- The corpora order the same way under every trained causal arm: Euclid lowest (2-layer 1.50 bits
  per character, 72.8% accuracy), then Herodotus (1.95), Tiny Shakespeare (2.48) and the Muqaddimah
  (2.79). Attention gains most over the bigram floor on Euclid (1.63 bits) and least on the
  Muqaddimah (0.88).
- Held-out against training loss (bits per character on as many training windows as held out)
  differs by corpus: on Herodotus the two agree (2-layer: 1.94 and 1.95), on Tiny Shakespeare and
  the Muqaddimah training is about 0.3 bits lower, and on Euclid 0.49 lower (1.01 against 1.50),
  the widest gap, growing with the model (ffn 0.17, 1-layer 0.44). The held-out part of each text is
  its last 10% (D4), so a gap mixes overfitting with how far that part differs from the rest; even
  the counted bigram is 0.16 bits worse held out on Euclid, and 0.02 better on Herodotus.

The naive predictions (the workplan's "Naive predictions", made before the study):

- "I'd expect it [Euclid] to reach a much lower loss than the other corpora": borne out, 0.45 bits
  per character under the next corpus (Herodotus) at 2 layers, and the largest gain from attention.
- "the masked versus unmasked comparison may show a sharper gap" on Euclid: not borne out. The gap
  is the smallest on Euclid (1.45 bits, against 1.90 to 2.74), since the unmasked arm reaches the
  same copying floor on every corpus and the gap is then the causal model's loss: the mask matters
  least where predicting from the past is easiest. Nor does the mask "matter more on structured
  text" by any other measure here.

| corpus | arm | rate | parameters | epoch 1 | epoch 5 | epoch 10 | accuracy | train | seconds per epoch |
|---|---|---|---|---|---|---|---|---|---|
| Tiny Shakespeare | unigram | counted | 65 | 4.829 | | | 14.89% | 4.769 | |
| | bigram | counted | 4225 | 3.581 | | | 26.98% | 3.534 | |
| | ffn | 0.0005 | 45825 | 3.614 ± 0.005 | 3.588 ± 0.004 | 3.591 ± 0.004 | 26.80% ± 0.11% | 3.537 ± 0.005 | 25.7 |
| | 1-layer | 0.004 | 62593 | 2.976 ± 0.013 | 2.666 ± 0.010 | 2.591 ± 0.008 | 47.24% ± 0.15% | 2.336 ± 0.010 | 53.4 |
| | 2-layer | 0.002 | 112577 | 2.977 ± 0.009 | 2.555 ± 0.013 | 2.476 ± 0.007 | 49.63% ± 0.21% | 2.172 ± 0.008 | 97.3 |
| | 2-layer-unmasked | 0.004 | 112577 | 0.062 ± 0.002 | 0.053 ± 0.001 | 0.051 ± 0.001 | 99.00% ± 0.02% | 0.047 ± 0.001 | 95.6 |
| Herodotus | unigram | counted | 76 | 4.310 | | | 17.63% | 4.303 | |
| | bigram | counted | 5776 | 3.344 | | | 32.16% | 3.362 | |
| | ffn | 0.001 | 47244 | 3.362 ± 0.003 | 3.353 ± 0.001 | 3.352 ± 0.001 | 32.11% ± 0.04% | 3.365 ± 0.002 | 34.9 |
| | 1-layer | 0.004 | 64012 | 2.490 ± 0.021 | 2.191 ± 0.008 | 2.119 ± 0.009 | 55.85% ± 0.16% | 2.128 ± 0.010 | 72.2 |
| | 2-layer | 0.002 | 113996 | 2.442 ± 0.017 | 2.037 ± 0.006 | 1.950 ± 0.007 | 59.14% ± 0.12% | 1.935 ± 0.007 | 131.7 |
| | 2-layer-unmasked | 0.002 | 113996 | 0.057 ± 0.001 | 0.048 ± 0.001 | 0.044 ± 0.001 | 99.12% ± 0.01% | 0.040 ± 0.001 | 128.8 |
| Muqaddimah | unigram | counted | 40 | 4.263 | | | 18.95% | 4.328 | |
| | bigram | counted | 1600 | 3.677 | | | 28.38% | 3.658 | |
| | ffn | 0.001 | 42600 | 3.697 ± 0.004 | 3.688 ± 0.004 | 3.688 ± 0.003 | 28.31% ± 0.16% | 3.660 ± 0.001 | 22.2 |
| | 1-layer | 0.004 | 59368 | 3.199 ± 0.014 | 2.948 ± 0.023 | 2.887 ± 0.010 | 44.76% ± 0.15% | 2.647 ± 0.016 | 47.4 |
| | 2-layer | 0.004 | 109352 | 3.203 ± 0.026 | 2.866 ± 0.012 | 2.794 ± 0.004 | 46.59% ± 0.22% | 2.508 ± 0.004 | 87.6 |
| | 2-layer-unmasked | 0.002 | 109352 | 0.064 ± 0.001 | 0.058 ± 0.001 | 0.056 ± 0.001 | 98.97% ± 0.01% | 0.049 ± 0.001 | 85.6 |
| Euclid | unigram | counted | 67 | 4.481 | | | 19.36% | 4.431 | |
| | bigram | counted | 4489 | 3.131 | | | 38.32% | 2.971 | |
| | ffn | 0.0005 | 46083 | 3.163 ± 0.006 | 3.144 ± 0.005 | 3.142 ± 0.005 | 38.45% ± 0.22% | 2.974 ± 0.009 | 18.8 |
| | 1-layer | 0.002 | 62851 | 2.018 ± 0.016 | 1.682 ± 0.015 | 1.622 ± 0.010 | 70.85% ± 0.18% | 1.181 ± 0.010 | 39.3 |
| | 2-layer | 0.002 | 112835 | 1.959 ± 0.021 | 1.567 ± 0.018 | 1.503 ± 0.015 | 72.81% ± 0.41% | 1.014 ± 0.013 | 72.2 |
| | 2-layer-unmasked | 0.004 | 112835 | 0.079 ± 0.006 | 0.058 ± 0.002 | 0.056 ± 0.001 | 99.21% ± 0.03% | 0.028 ± 0.001 | 52.4 |

Held-out bits per character, mean ± standard deviation over the 5 seeds; accuracy is per token
after epoch 10, and "train" the bits per character on the first training windows then. A difference
quoted between arms is the mean of the per-seed differences, lower on every seed unless said (the
arms share seeds, so their shuffles are the same). Seconds per epoch are the mean over the seeds and
epochs, 6 jobs at once (2.4x one job alone); the last jobs ran with fewer beside them, so Euclid's
2-layer-unmasked, among the last, shows less. The floors are counted, not trained, so they have one
value. The per-epoch tables and tune's are in the stage's PR.
"""

from indrajala_ml.studies import sequence_study

if __name__ == "__main__":
    sequence_study.main(description=__doc__)
