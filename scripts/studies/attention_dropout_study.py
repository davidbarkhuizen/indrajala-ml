"""
The attention-dropout study (the attention-dropout workplan, stage 8, D8): does dropout close the
sequence study's train/held-out gaps on Euclid (Heath), Tiny Shakespeare and Herodotus
(Rawlinson), and is the gap overfitting or the held-out text's shift? numpy, under
indrajala_ml/studies/attention_dropout_study.py's protocol.

    python scripts/studies/attention_dropout_study.py time --corpus herodotus-rawlinson --arm "both 0.2"
    python scripts/studies/attention_dropout_study.py sweep --epochs 10 --seeds 5 --out sweep.json

Five arms at the sequence study's 2-layer model (control; attention 0.1, `Attention(dropout)`;
residual 0.1, `Dropout` on the embeddings and each block's residual branches; both 0.1; both 0.2) x
three corpora x two splits (contiguous: the last 10% held out, as the sequence study; spread: each
of 100 blocks' last tenth) x 5 seeds x 10 epochs. Adam at batch 32, one rate for every arm, 0.002
(the sequence study's tuned 2-layer rate on Euclid, Tiny Shakespeare and Herodotus),
OPENBLAS_NUM_THREADS=1, 6 workers.

Findings, on jebel (i7-9700K), nothing else running. One epoch of the slowest cell, Herodotus
`both 0.2`, took 61.0 s alone (`time`; 2-layer without dropout 54.4 s), and the sweep 8.5 hours.
Every run stayed finite.

- The run is valid: the control on the contiguous split reproduces the sequence study's 2-layer
  rows to every printed digit on all three corpora (Euclid 1.503 bits, Tiny Shakespeare 2.476,
  Herodotus 1.950 held out; the same train bits and accuracies).
- The gaps are mostly the text's shift, not overfitting. On the spread split the control's gap
  falls from 0.489 to 0.036 bits per character on Euclid (93% of it was the shift) and from 0.303
  to 0.068 on Tiny Shakespeare (78%), while its training bits stay the same (Euclid 1.014 and
  1.018): the last tenth of each text is harder than the rest, not unseen. On Herodotus the two
  splits' gaps are both small (0.014 contiguous, 0.039 spread).
- Dropout narrows every gap, but by raising the training loss, not by lowering the held-out one.
  Residual and both cost 0.057 to 0.354 bits held out against the control (means), higher on every
  seed, corpus and split; both 0.2 costs the most (0.250 to 0.354). At this model's size and 10
  epochs there is little overfitting for dropout to remove.
- The one gain is attention dropout on Euclid's contiguous split: 0.023 ± 0.009 bits lower held
  out, on every seed, its gap 0.432 against 0.489. It doesn't hold on the spread split (0.018
  higher, every seed) nor on Tiny Shakespeare (+0.010 contiguous, lower on 1 seed of 5; +0.025
  spread), so it helps across the shift into Euclid's last tenth, not against overfitting.
  Attention 0.1 is the cheapest arm everywhere else (+0.010 to +0.025).
- Herodotus, where dropout should only cost, does: every arm is higher held out than the control
  on every seed of both splits (+0.020 to +0.354), and the dropout arms' gaps go negative on the
  contiguous split (to -0.036 at both 0.2): measured without dropout, its training windows are
  then harder than its held-out text.
- No arm had converged, and the dropout arms least: held-out bits fell 0.044 to 0.081 over the
  last three epochs under dropout, 0.014 to 0.047 under the control. Longer runs, and a rate tuned
  per arm (dropout may prefer a higher one), could narrow the costs; whether they turn into gains
  is open.
- Cost: attention dropout adds 10% to an epoch, residual 6-7%, both 17%, the same on every corpus
  (seconds per epoch against the control, 6 jobs at once). Attention's mask is drawn over every
  head's `T x T` weights, the residual's over `T x d`.

The questions (the workplan's D8): which mechanism closes how much of each gap: residual more than
attention (Euclid contiguous 0.088 against 0.057), both 0.1 0.105, both 0.2 the same as both 0.1
there and more on the other two corpora, every one by fitting the training text worse. Herodotus: dropout only
costs. Euclid: its gap on the spread split is far smaller, so the gap is the shift.

Held-out bits per character after epochs 1, 5 and 10, mean ± standard deviation over the 5 seeds;
accuracy is per token after epoch 10, "train" the bits per character on as many of the first
training windows as held out, both evaluated without dropout; "gap" held out minus train, and
"against control" the mean of the per-seed differences in final held-out bits (the arms share
seeds, so their shuffles and initial weights are the same). Seconds per epoch are the mean over the
seeds and epochs with 6 jobs at once; the last jobs ran with fewer beside them, so Herodotus's
spread `both 0.2`, among the last, shows less.

euclid-heath, contiguous

| arm | epoch 1 | epoch 5 | epoch 10 | accuracy | train | gap | against control | seconds per epoch |
|---|---|---|---|---|---|---|---|---|
| control | 1.959 ± 0.021 | 1.567 ± 0.018 | 1.503 ± 0.015 | 72.81% ± 0.41% | 1.014 ± 0.013 | 0.489 ± 0.011 |  | 73.0 |
| attention 0.1 | 1.997 ± 0.015 | 1.568 ± 0.009 | 1.481 ± 0.010 | 72.79% ± 0.27% | 1.048 ± 0.009 | 0.432 ± 0.012 | -0.023 ± +0.009 | 80.6 |
| residual 0.1 | 2.242 ± 0.013 | 1.673 ± 0.007 | 1.560 ± 0.007 | 71.31% ± 0.26% | 1.159 ± 0.006 | 0.401 ± 0.009 | +0.057 ± +0.019 | 77.8 |
| both 0.1 | 2.302 ± 0.027 | 1.703 ± 0.015 | 1.584 ± 0.009 | 70.71% ± 0.12% | 1.200 ± 0.006 | 0.384 ± 0.015 | +0.081 ± +0.015 | 85.4 |
| both 0.2 | 2.590 ± 0.040 | 1.906 ± 0.024 | 1.754 ± 0.009 | 67.50% ± 0.19% | 1.369 ± 0.008 | 0.384 ± 0.010 | +0.250 ± +0.022 | 85.3 |

euclid-heath, spread

| arm | epoch 1 | epoch 5 | epoch 10 | accuracy | train | gap | against control | seconds per epoch |
|---|---|---|---|---|---|---|---|---|
| control | 1.607 ± 0.014 | 1.137 ± 0.007 | 1.054 ± 0.006 | 78.21% ± 0.10% | 1.018 ± 0.006 | 0.036 ± 0.002 |  | 72.9 |
| attention 0.1 | 1.680 ± 0.007 | 1.167 ± 0.010 | 1.072 ± 0.008 | 77.67% ± 0.13% | 1.063 ± 0.011 | 0.010 ± 0.007 | +0.018 ± +0.006 | 81.1 |
| residual 0.1 | 1.985 ± 0.030 | 1.283 ± 0.004 | 1.170 ± 0.009 | 75.73% ± 0.17% | 1.168 ± 0.008 | 0.002 ± 0.004 | +0.116 ± +0.012 | 78.0 |
| both 0.1 | 2.072 ± 0.037 | 1.330 ± 0.003 | 1.210 ± 0.006 | 74.80% ± 0.12% | 1.212 ± 0.007 | -0.001 ± 0.004 | +0.156 ± +0.007 | 85.4 |
| both 0.2 | 2.472 ± 0.049 | 1.548 ± 0.012 | 1.391 ± 0.005 | 71.25% ± 0.18% | 1.382 ± 0.006 | 0.009 ± 0.002 | +0.337 ± +0.003 | 85.3 |

tinyshakespeare, contiguous

| arm | epoch 1 | epoch 5 | epoch 10 | accuracy | train | gap | against control | seconds per epoch |
|---|---|---|---|---|---|---|---|---|
| control | 2.977 ± 0.009 | 2.555 ± 0.013 | 2.476 ± 0.007 | 49.63% ± 0.21% | 2.172 ± 0.008 | 0.303 ± 0.005 |  | 99.1 |
| attention 0.1 | 2.994 ± 0.009 | 2.583 ± 0.010 | 2.485 ± 0.004 | 48.98% ± 0.17% | 2.212 ± 0.007 | 0.274 ± 0.009 | +0.010 ± +0.010 | 109.5 |
| residual 0.1 | 3.133 ± 0.013 | 2.702 ± 0.021 | 2.587 ± 0.019 | 46.81% ± 0.28% | 2.341 ± 0.018 | 0.246 ± 0.008 | +0.111 ± +0.020 | 104.9 |
| both 0.1 | 3.159 ± 0.008 | 2.730 ± 0.014 | 2.612 ± 0.007 | 46.23% ± 0.08% | 2.372 ± 0.009 | 0.239 ± 0.010 | +0.136 ± +0.011 | 115.9 |
| both 0.2 | 3.333 ± 0.021 | 2.898 ± 0.016 | 2.790 ± 0.011 | 42.98% ± 0.26% | 2.577 ± 0.014 | 0.212 ± 0.012 | +0.314 ± +0.014 | 115.6 |

tinyshakespeare, spread

| arm | epoch 1 | epoch 5 | epoch 10 | accuracy | train | gap | against control | seconds per epoch |
|---|---|---|---|---|---|---|---|---|
| control | 2.865 ± 0.011 | 2.347 ± 0.006 | 2.244 ± 0.007 | 52.89% ± 0.15% | 2.176 ± 0.007 | 0.068 ± 0.005 |  | 99.2 |
| attention 0.1 | 2.891 ± 0.009 | 2.380 ± 0.011 | 2.269 ± 0.003 | 52.29% ± 0.11% | 2.214 ± 0.005 | 0.055 ± 0.008 | +0.025 ± +0.005 | 108.5 |
| residual 0.1 | 3.085 ± 0.019 | 2.516 ± 0.010 | 2.380 ± 0.017 | 50.11% ± 0.29% | 2.340 ± 0.014 | 0.041 ± 0.006 | +0.136 ± +0.016 | 103.9 |
| both 0.1 | 3.116 ± 0.021 | 2.553 ± 0.009 | 2.405 ± 0.011 | 49.52% ± 0.20% | 2.370 ± 0.011 | 0.036 ± 0.005 | +0.161 ± +0.007 | 115.9 |
| both 0.2 | 3.321 ± 0.017 | 2.757 ± 0.007 | 2.581 ± 0.027 | 46.26% ± 0.50% | 2.555 ± 0.025 | 0.026 ± 0.005 | +0.337 ± +0.023 | 114.9 |

herodotus-rawlinson, contiguous

| arm | epoch 1 | epoch 5 | epoch 10 | accuracy | train | gap | against control | seconds per epoch |
|---|---|---|---|---|---|---|---|---|
| control | 2.442 ± 0.017 | 2.037 ± 0.006 | 1.950 ± 0.007 | 59.14% ± 0.12% | 1.935 ± 0.007 | 0.014 ± 0.005 |  | 133.3 |
| attention 0.1 | 2.493 ± 0.024 | 2.064 ± 0.003 | 1.970 ± 0.002 | 58.63% ± 0.13% | 1.976 ± 0.005 | -0.005 ± 0.004 | +0.021 ± +0.006 | 146.6 |
| residual 0.1 | 2.680 ± 0.016 | 2.191 ± 0.010 | 2.078 ± 0.006 | 56.49% ± 0.11% | 2.099 ± 0.008 | -0.021 ± 0.003 | +0.128 ± +0.011 | 141.2 |
| both 0.1 | 2.734 ± 0.025 | 2.229 ± 0.008 | 2.109 ± 0.003 | 55.88% ± 0.14% | 2.133 ± 0.001 | -0.024 ± 0.003 | +0.159 ± +0.009 | 156.0 |
| both 0.2 | 2.970 ± 0.037 | 2.417 ± 0.015 | 2.289 ± 0.018 | 52.61% ± 0.29% | 2.325 ± 0.018 | -0.036 ± 0.002 | +0.339 ± +0.016 | 155.7 |

herodotus-rawlinson, spread

| arm | epoch 1 | epoch 5 | epoch 10 | accuracy | train | gap | against control | seconds per epoch |
|---|---|---|---|---|---|---|---|---|
| control | 2.476 ± 0.014 | 2.064 ± 0.007 | 1.975 ± 0.009 | 58.69% ± 0.23% | 1.936 ± 0.013 | 0.039 ± 0.004 |  | 133.5 |
| attention 0.1 | 2.511 ± 0.006 | 2.092 ± 0.004 | 1.995 ± 0.001 | 58.24% ± 0.10% | 1.971 ± 0.003 | 0.024 ± 0.003 | +0.020 ± +0.009 | 146.4 |
| residual 0.1 | 2.709 ± 0.013 | 2.216 ± 0.010 | 2.106 ± 0.013 | 56.05% ± 0.26% | 2.093 ± 0.014 | 0.013 ± 0.002 | +0.131 ± +0.014 | 141.1 |
| both 0.1 | 2.754 ± 0.018 | 2.258 ± 0.007 | 2.146 ± 0.009 | 55.23% ± 0.16% | 2.137 ± 0.009 | 0.009 ± 0.002 | +0.171 ± +0.010 | 155.9 |
| both 0.2 | 2.975 ± 0.032 | 2.447 ± 0.009 | 2.329 ± 0.003 | 51.68% ± 0.12% | 2.332 ± 0.004 | -0.003 ± 0.002 | +0.354 ± +0.010 | 137.9 |
"""

from indrajala_ml.studies import attention_dropout_study

if __name__ == "__main__":
    attention_dropout_study.main(description=__doc__)
